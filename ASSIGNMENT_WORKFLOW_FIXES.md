# Assignment Workflow Fixes

## Problem Summary
The assignment workflow had several critical issues:
1. **Missing assignee email persistence** in `bid_assign` table
2. **No synchronization** between `go_bids.state` and assigned department
3. **Role dashboards querying wrong table** (only `go_bids` instead of `bid_assign`)
4. **Master dashboard timeline not syncing** with assignments

## Fixes Implemented

### 1. Database Schema Updates
**File:** `app_v2.py` (table creation)

```sql
-- Added assignee_email column to bid_assign table
CREATE TABLE IF NOT EXISTS bid_assign (
    a_id INT AUTO_INCREMENT PRIMARY KEY,
    g_id INT,
    b_name VARCHAR(100),
    in_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    due_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state VARCHAR(100),
    scope VARCHAR(100),
    type VARCHAR(100),
    company TEXT,
    depart TEXT,
    person_name TEXT,
    assignee_email VARCHAR(100),  -- NEW COLUMN
    status TEXT,
    value INT
)
```

### 2. Assignment Function Enhancement
**File:** `app_v2.py` (`dbm_assign_go` function)

**Key Changes:**
- **Department-to-Stage Mapping**: Automatically maps assigned department to correct stage
- **State Synchronization**: Updates `go_bids.state` to match assigned department
- **Email Persistence**: Stores assignee email in `bid_assign.assignee_email`
- **Comprehensive Logging**: Logs all assignment actions
- **Real-time Updates**: Emits Socket.IO events for master dashboard sync

```python
# Department to stage mapping
dept_to_stage = {
    'business dev': 'business',
    'business': 'business',
    'design': 'design',
    'operations': 'operations',
    'site manager': 'engineer',
    'engineer': 'engineer'
}

# Update go_bids.state to match assigned department
cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, g_id))

# Store assignee email
cur.execute("""
    INSERT INTO bid_assign (..., assignee_email, ...)
    VALUES (..., %s, ...)
""", (..., email_to, ...))
```

### 3. Team Dashboard Updates
**File:** `app_v2.py` (`team_dashboard` function)

**Key Changes:**
- **Query Source**: Changed from `go_bids` to `bid_assign` (assigned bids only)
- **Assignee Information**: Shows person name and email
- **Department Filtering**: Only shows bids assigned to the team's department

```python
# Query assigned bids instead of all go_bids
cur.execute("""
    SELECT ba.a_id AS id,
           ba.b_name AS name,
           ba.assignee_email,
           ba.person_name,
           ba.depart,
           ...
    FROM bid_assign ba
    WHERE LOWER(COALESCE(ba.state, 'analyzer')) = %s
""", (current_stage,))
```

### 4. Role Dashboard Updates
**File:** `app_v2.py` (`role_dashboard` function)

**Key Changes:**
- **Query Source**: Changed from `go_bids` to `bid_assign`
- **Assignee Display**: Shows assignee email and department
- **Consistent Data**: All role dashboards now show assigned bids only

### 5. Master Dashboard Synchronization
**File:** `app_v2.py` (assignment function) + `templates/master_dashboard.html`

**Key Changes:**
- **Socket.IO Emissions**: Assignment actions emit real-time updates
- **Timeline Sync**: Master dashboard refreshes when assignments are made
- **Activity Logging**: All assignment actions appear in live activity log

```python
# Emit real-time update for master dashboard
socketio.emit('master_update', {
    'bid': {
        'id': g_id,
        'name': bid_name,
        'current_stage': new_stage,
        'assigned_to': person_name,
        'department': depart
    },
    'log': {...},
    'assignment': True
})
```

### 6. UI Enhancements
**File:** `templates/team_dashboard.html`

**Key Changes:**
- **Assignee Column**: Added assignee name and email display
- **Better Information**: Shows who is assigned to each bid
- **Responsive Design**: Maintains clean layout with new information

## Workflow Now Works As Follows

### 1. Admin Assignment Process
1. Admin goes to Database Management → GO Bids
2. Clicks "Assign" on a bid
3. Fills in:
   - **Department** (business dev, design, operations, site manager)
   - **Person Name** (assignee name)
   - **Email** (assignee email)
4. System automatically:
   - Updates `go_bids.state` to match department
   - Creates/updates `bid_assign` record with assignee info
   - Logs the assignment action
   - Emits real-time update to master dashboard
   - Sends email notification (if email provided)

### 2. Team Dashboard Display
1. Each team dashboard shows only bids assigned to their department
2. Displays assignee information (name and email)
3. Shows progress, status, and due dates
4. Allows stage advancement with proper logging

### 3. Master Dashboard Sync
1. Timeline trackers reflect current bid states
2. Live activity log shows all assignment actions
3. Real-time updates when assignments are made
4. Comprehensive overview of all bid progress

## Testing

### Manual Testing
1. **Start the application**: `python app_v2.py`
2. **Login as admin** and go to Database Management
3. **Assign a bid** to different departments
4. **Check team dashboards** to verify bids appear
5. **Verify master dashboard** updates in real-time

### Automated Testing
Run the test suite: `python test_assignment_workflow.py`

## Database Verification

### Check Assignment Data
```sql
-- View all assignments
SELECT ba.*, gb.state as go_bids_state 
FROM bid_assign ba 
LEFT JOIN go_bids gb ON ba.g_id = gb.g_id;

-- Check state synchronization
SELECT g_id, b_name, state, depart 
FROM bid_assign 
WHERE state != depart;
```

### Check Logs
```sql
-- View assignment logs
SELECT * FROM logs 
WHERE action LIKE '%assigned%' 
ORDER BY timestamp DESC;
```

## Benefits

1. **Complete Assignment Tracking**: Every assignment is logged and tracked
2. **State Synchronization**: `go_bids.state` always matches assigned department
3. **Team-Specific Views**: Each team sees only their assigned bids
4. **Real-Time Updates**: Master dashboard stays in sync with assignments
5. **Audit Trail**: Complete history of all assignment actions
6. **Email Notifications**: Assignees are notified of new assignments

## Future Enhancements

1. **Bulk Assignment**: Assign multiple bids at once
2. **Assignment History**: Track assignment changes over time
3. **Workload Balancing**: Distribute bids evenly across team members
4. **Assignment Templates**: Pre-configured assignment patterns
5. **Advanced Notifications**: SMS, Slack, or other notification methods
