# Advance Workflow Implementation

## Overview
Implemented the complete advance workflow where sub-dashboards show bids based on `go_bids.state` and clicking "Advance" moves bids to the next team's dashboard immediately.

## Implementation Details

### Step 1: Query by go_bids.state (not bid_assign.depart)

**File:** `app_v2.py` (`role_dashboard` function)

```python
# Query by go_bids.state and join assignee info
role_to_stage = {
    'business dev': 'business',
    'business': 'business', 
    'bdm': 'business',
    'design': 'design',
    'operations': 'operations',
    'site manager': 'engineer',
    'site_manager': 'engineer'
}
current_stage_for_role = role_to_stage.get((current_user.role or 'member').lower())

sql = """
    SELECT gb.g_id AS id,
           gb.b_name AS name,
           gb.company, 
           gb.due_date,
           COALESCE(gb.scoring, 0) AS progress,
           LOWER(COALESCE(gb.state, 'analyzer')) AS current_stage,
           ba.person_name,
           ba.assignee_email AS person_email
    FROM go_bids gb
    LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
    {where}
    ORDER BY gb.due_date ASC
"""
where = "WHERE LOWER(COALESCE(gb.state, 'analyzer')) = %s" if current_stage_for_role else ""
cur.execute(sql.format(where=where), (current_stage_for_role,) if current_stage_for_role else ())
```

**Key Changes:**
- Filters by `go_bids.state` instead of `bid_assign.depart`
- LEFT JOINs `bid_assign` for assignee information
- Maintains proper stage mapping for role-to-stage conversion

### Step 2: Advance API Updates Both Tables

**File:** `app_v2.py` (`update_stage` function)

```python
# Validate stage
allowed = {'analyzer', 'business', 'design', 'operations', 'engineer', 'handover'}
if new_stage not in allowed:
    return jsonify({'error': 'invalid stage'}), 400

# 2a) Update go_bids.state
cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, bid_id))

# 2b) Upsert assignment so the next team still sees it
cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (bid_id,))
row = cur.fetchone()
if row:
    cur.execute("UPDATE bid_assign SET depart=%s, state=%s, status='pending' WHERE g_id=%s",
                (new_stage, new_stage, bid_id))
else:
    cur.execute("""
        INSERT INTO bid_assign (g_id, b_name, in_date, due_date, state, scope, type, company, depart,
                               person_name, assignee_email, status, value)
        SELECT g_id, b_name, in_date, due_date, state, scope, type, company, %s, '', '', 'pending',
               COALESCE(scoring, 0)
        FROM go_bids WHERE g_id=%s
    """, (new_stage, bid_id))
```

**Key Changes:**
- Validates stage against allowed set
- Updates `go_bids.state` to new stage
- Upserts `bid_assign` record for next team
- Maintains data consistency between tables

### Step 3: Team Dashboard Updates

**File:** `app_v2.py` (`team_dashboard` function)

```python
# Query by go_bids.state (not bid_assign.depart) and join assignee info
cur.execute("""
    SELECT gb.g_id AS id,
           gb.b_name AS name,
           gb.company, 
           gb.due_date,
           COALESCE(gb.scoring, 0) AS progress,
           LOWER(COALESCE(gb.state, 'analyzer')) AS current_stage,
           ba.person_name,
           ba.assignee_email AS person_email,
           ba.depart,
           wps.pr_completion_status AS work_status,
           wbr.closure_status AS project_status,
           wbr.work_progress_status AS work_progress_status,
           wlr.result AS wl_result
    FROM go_bids gb
    LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
    LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
    LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
    LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
    WHERE LOWER(COALESCE(gb.state, 'analyzer')) = %s
    ORDER BY gb.due_date ASC
""", (current_stage,))
```

**Key Changes:**
- Uses `go_bids.state` for filtering instead of `bid_assign.depart`
- Joins all related tables for comprehensive bid information
- Maintains consistent data structure across all dashboards

### Step 4: Frontend Action

**File:** `templates/team_dashboard.html`

```javascript
function advanceStage(bidId, nextStage) {
    if (!nextStage || nextStage === 'null' || nextStage === 'undefined') {
        alert('No next stage available - this is the final stage');
        return;
    }
    
    fetch(`/api/update_stage/${bidId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: nextStage })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message briefly then reload
            const notification = document.createElement('div');
            notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50';
            notification.textContent = data.success;
            document.body.appendChild(notification);
            
            // Reload page to move bid to next team's dashboard
            setTimeout(() => {
                location.reload();
            }, 1000);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error updating stage');
    });
}
```

**Key Changes:**
- Validates nextStage before making request
- Shows success notification
- Reloads page to move bid to next team's dashboard
- Handles errors gracefully

## Stage Flow

The complete stage progression:

```
analyzer → business → design → operations → engineer → handover
```

### Team Dashboard Mapping

- **Business Dev** (`/dashboard/business`) → Shows bids with `state='business'`
- **Design** (`/dashboard/design`) → Shows bids with `state='design'`
- **Operations** (`/dashboard/operations`) → Shows bids with `state='operations'`
- **Site Engineer** (`/dashboard/engineer`) → Shows bids with `state='engineer'`

## Data Flow

1. **Bid Assignment**: Admin assigns bid to department → `go_bids.state` updated
2. **Team Dashboard**: Shows bids where `go_bids.state` matches team stage
3. **Advance Action**: User clicks "Advance" → API updates both tables
4. **Immediate Update**: Page reloads → bid appears on next team's dashboard

## Benefits

1. **Real-time Movement**: Bids move between teams immediately
2. **Data Consistency**: Both `go_bids` and `bid_assign` stay in sync
3. **Clear Workflow**: Each team sees only their stage's bids
4. **Audit Trail**: All movements are logged and tracked
5. **User Experience**: Simple advance button with immediate feedback

## Testing

Run the comprehensive test suite:
```bash
python test_advance_workflow.py
```

### Manual Testing Steps

1. **Start Flask app**: `python app_v2.py`
2. **Login as admin** and assign a bid to 'business' department
3. **Check `/dashboard/business`** - should show the bid
4. **Click 'Advance' button** - bid should move to `/dashboard/design`
5. **Verify bid appears** on design dashboard
6. **Repeat for all stages**: design → operations → engineer → handover

## Status
✅ **COMPLETE** - The advance workflow is fully implemented and functional. Sub-dashboards now show bids based on `go_bids.state` and clicking "Advance" moves bids to the next team's dashboard immediately.
