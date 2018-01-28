#%% Goal: Plan issues ETAs from order and efforts
import numpy as np
import pandas as pd
from time import time
from statusClassifier import CLOSED, isProgressStatus
from statusOrder import orderStatuses
from eventFeatures import MINIMUM_HOURS_EVENT, MAXIMUM_HOURS_EVENT
from timeUtils import workDayHours, dateAfterManHours, dateBeforeManHours, toIso
daysInWeek = 5

VELOCITY_WEEKS = 5  # Recent period to estimate the velocity
MARGIN_WEEKS = 1    # Recent period to ignore (too fresh, not finished)

maxTimestampStr = '2111-11-11T11:11:11Z'
maxTimestamp = pd.Timestamp(maxTimestampStr)
maxId = 2**31
maxString = "zzz"

def listUnique(series):
    return list(series.unique())

# Velocity and parallelism per project

def measureProjectsVelocityPrecise(eventReprs):
    """ Return progress rate per day spent on each project (including parallelism).
        Outdated version based on hoursDelay of events during progress.
    """
    events = eventReprs["event"]
    # Events that count as progress
    isProgress = isProgressStatus(events["status"])

    # Recent events only
    now = events["dt"].max()
    start = now - pd.Timedelta(weeks=VELOCITY_WEEKS + MARGIN_WEEKS)
    end = now - pd.Timedelta(weeks=MARGIN_WEEKS)
    isRecent = events["dt"].between(start, end)

    # Exclude bulk actions
    period = pd.Grouper(freq="1H", key="dt")
    actionsByPeriod = events.groupby([period, "author", "action"]).author.transform("size")
    isNotBulk = actionsByPeriod < 25

    prog = events.loc[isProgress & isRecent & isNotBulk, ["project", "author", "hoursDelay"]]
    prog["hoursDelay"] = prog["hoursDelay"].clip(MINIMUM_HOURS_EVENT, MAXIMUM_HOURS_EVENT) / events["numSimultaneous"]

    projectVelocity = prog.groupby(["project", "author"])["hoursDelay"].sum() #.agg(["mean","count","sum"])
    projectVelocity /= VELOCITY_WEEKS * daysInWeek * workDayHours
    return projectVelocity


def measureProjectsVelocityRobust(eventReprs):
    " Return progress rate spent on each project (including parallelism). Based on the last event of progress. "

    # Recent events only
    now = eventReprs[("event", "dt")].max()
    start = now - pd.Timedelta(weeks=VELOCITY_WEEKS + MARGIN_WEEKS)
    end = now - pd.Timedelta(weeks=MARGIN_WEEKS)

    # Find recent end of progress
    recentlyClosed = eventReprs[("targets", "endDate")].between(start, end)
    if recentlyClosed.sum() < 50:
        print("Warning: very few recent issues (",recentlyClosed.sum(),"), using all instead.")
        recentlyClosed[:] = True

    isEnd = (eventReprs[("event","dt")] == eventReprs[("targets", "endDate")])
    isStatus = (eventReprs[("event","field")] == "status")
    recentEfforts = eventReprs.loc[
        recentlyClosed & isEnd & isStatus,
        [("event","project"), ("event","author"), ("targets", "effortSpent")]
    ]
    assert recentEfforts.groupby("key_i").size().max() == 1, "End-of-progress should be unique per key"
    #recentEfforts.index = recentEfforts.index.get_level_values("key_i")
    recentEfforts.columns = ["project", "author", "effortSpent"]

    g = recentEfforts.groupby("project")
    perProject = pd.concat([
        g["author"].agg(listUnique),
        g["effortSpent"].agg(["count", "median", "sum"]),
    ], axis=1)
    perProject["sum"] /= VELOCITY_WEEKS * daysInWeek * workDayHours # As a rate
    perProject.columns = [
        "People_progressing",
        "Measure_issues_count", "Measure_effort_hours_median", "Rate_of_progress"]

    # TODO: Convert to author display names. Sort maybe?
    return perProject


# Order the sprints
def orderSprints(rawSprints):
    sprints = pd.DataFrame.from_dict(rawSprints, "index")
    sprints["state"].fillna(maxString, inplace=True)
    sprints["startDate"].fillna(maxTimestampStr, inplace=True)
    sprints.sort_values([
        "state",        # active < future < NaN
        "startDate",    # Earlier start should mean earlier work
        "id",           # Proxy for creation time, lower ID comes likely earlier
    ], inplace=True)
    sprints["order"] = range(len(sprints))
    return sprints


# Ranking issues by likely order of start

def predictIssueOrder(issues, statusOrder, rawSprints):
    print("Ordered statuses:", list(statusOrder.index))

    if rawSprints:
        sprints = orderSprints(rawSprints)
        sprintOrder = sprints["order"]
        print("Sprints to come:\n", sprints[["name","state"]].query("state!='closed'"))
    else:
        sprintOrder = {}
        print("No sprints")

    #print("Status categories:\n", issues.statusClass.value_counts().sort_index())
    print("Priorities:\n", issues.groupby("priorityId").priority.value_counts(dropna=False).sort_index())

    # Must replace the NaN to sort them at the end (unlike documented)
    sortValues = pd.concat([
        #issues["statusClass"].fillna(maxString),    # closed < progress < todo < unknown
        issues["sprintId"].map(sprintOrder).fillna(maxId),  # Active and started sprints come earlier
        -issues["status"].map(statusOrder).fillna(maxId),    # Common order of statuses, "closed" first
        issues["startDate"].fillna(maxTimestamp),   # Already started < NaN
        issues["priorityId"].fillna(maxId),         # Lower ID is more important
        issues["rank"].fillna(maxString),           # Lower rank is at the top of the list
        #("updated", False),                        # Recently updated is more likely to be relevant
    ], axis="columns")

    sortedIndex = sortValues.sort_values(list(sortValues.columns)).index
    return issues.reindex(sortedIndex)


# Run through issues, accumulating efforts and increasing dates

def allocateNaive():
    startsInHours = (effortLeft.shift(1).fillna(0).cumsum() / vel).clip(0, 1e4)


def allocateLanes(effortLeft, velocity):
    " Plan issues using a lanes/max parallelism model. "
    nLanes = int(np.ceil(velocity))                         # Max parallelism.
    laneNextEmptyTime = pd.Series(np.zeros(nLanes))         # End times of last allocated issues.
    allocs = pd.DataFrame(
        0.0,
        columns=["startTime", "lane"],
        index=effortLeft.index)    # Output start hours of issues.

    for key, ef in list(effortLeft.items()):            # Allocate in order of priority.
        lane = laneNextEmptyTime.argmin()
        nextStart = laneNextEmptyTime.loc[lane]         # First lane to become available.
        allocs.loc[key] = (nextStart, lane)             # Start the issue.
        laneNextEmptyTime.loc[lane] = nextStart + ef    # Make the lane busy for a while.
        #print("%16s  %1i  %.1f  %.1f" % (key, lane, nextStart, ef))

    allocs["startTime"] *= nLanes / velocity                # Correct to match the real velocity.
    allocs["endTime"] = allocs["startTime"] + effortLeft    # The duration of issues is used as predicted.
    return allocs


def measurePastDates(eventReprs):
    return eventReprs.targets.groupby("key_i").agg({
        "startDate":"last", "endDate":"last", "closeDate":"last",
    })


def makeWorkDates(etaPreds, efforts):
    df = pd.DataFrame({
        "endDate": etaPreds["endDate"],
        "effortTotal": efforts.effortSpent + efforts.effortLeft,
    })
    # Estimate the real start of work with: start = end - effort
    workDate = df.apply(
        lambda row: dateBeforeManHours(
            row.effortTotal,
            row.endDate
        ), axis=1)
    # Use measured startDate if the estimate is close. If NaT, stay NaT.
    preferStart = (workDate - etaPreds["startDate"]) < pd.Timedelta('1 day')
    workDate.where(~preferStart, etaPreds["startDate"], inplace=True) # Must use inplace (pandas bug #9065).
    workDate.name = "workDate"
    return workDate


def learnPlan(issues, eventReprs, efforts, rawSprints):
    scriptStartTime = time()

    # Prepare issues that already started
    measures = measurePastDates(eventReprs)

    statusOrder = orderStatuses(eventReprs["event"])

    # Plan issues that are not closed
    openIssues = issues.query("statusClass != '%s'" % CLOSED).join(measures["startDate"] )
    plan = predictIssueOrder(openIssues, statusOrder, rawSprints).join(efforts.effortLeft.dropna(), how="inner")

    # Future plan stats
    projectStatsPlan = plan.groupby("project").effortLeft.agg(["count", "median", "max"])
    projectStatsPlan.columns = ["Plan_issues_count", "Effort_left_hours_median", "Effort_left_hours_max"]

    # Past measures stats
    projectStatsMeasure = measureProjectsVelocityRobust(eventReprs)

    now = eventReprs[("event", "dt")].max()
    preds = []

    for project, keys in plan.groupby("project").groups.items():
        velocity = projectStatsMeasure["Rate_of_progress"].get(project, 0)
        print("Velocity for project", project, ":", velocity)
        if not (velocity > 0.1):
            print("Velocity is too small (<0.1), skipping")
            continue

        effortLeft = plan.loc[keys, "effortLeft"]
        allocs = allocateLanes(effortLeft, velocity)

        preds.append(pd.DataFrame({
            "startDate": allocs["startTime"].apply(dateAfterManHours, args=[now]),
            "endDate": allocs["endTime"].apply(dateAfterManHours, args=[now]),
            "lane": allocs["lane"],
        }))

    preds = pd.concat(preds)
    preds["closeDate"] = preds["endDate"]   # Not predicting end->close delay

    # Combine with the known start and end dates
    preds = measures.combine_first(preds)

    # Estimate a reasonnable start of work, if different than startDate
    preds["workDate"] = makeWorkDates(preds, efforts)

    # Output metadata
    projectStats = pd.concat([
        projectStatsMeasure,
        projectStatsPlan,
    ], axis="columns").fillna(0)

    meta = {
        # Aggregates for all projects
        "Compute_minutes": (time() - scriptStartTime) / 60,
        "Plan_issues_count": len(preds),
        "Plan_first_start_date": toIso(preds["startDate"].min()),
        "Plan_last_end_date": toIso(preds["endDate"].max()),
    }
    return preds, meta, projectStats
