# Comprehensive Workflow Implementation

## Overview
Implemented a complete workflow system with auto-assignment, stage selectors, comprehensive logging, and real-time updates across all dashboards.

## Key Features Implemented

### 1. Auto-Assignment System
**GO bids automatically move to Business stage:**
- When a bid is marked "GO" in Bid Analyzer, it auto-creates/updates `go_bids` with `state='business'`
- Updates `bid_assign` with `depart='business'` if present
- Works in both create and update operations

### 2. Stage Selector System
**Replaced Advance buttons with flexible stage selectors:**
- Each team dashboard shows a dropdown with all available stages
- Teams can move bids forward or backward through the pipeline
- Excludes current stage from options
- Clean, intuitive interface

### 3. Comprehensive Logging System
**Logs all major operations:**
- **Login/Logout**: User authentication events
- **CRUD Operations**: Create, update, delete operations
- **Stage Changes**: All bid stage transitions
- **Assignments**: Auto-assignments and manual assignments
- **Real-time Streaming**: Admin sees live logs via Socket.IO

### 4. Real-Time Updates
**Master dashboard updates live:**
- Progress bars update instantly
- Status texts refresh automatically
- Timeline trackers move dynamically
- No page reloads needed

## Technical Implementation

### Backend Changes

#### Constants and Helpers
```python
PIPELINE = ['analyzer', 'business', 'design', 'operations', 'engineer', 'handover']
LABELS = {
    'analyzer': 'BID Analyzer',
    'business': 'Business Development', 
    'design': 'Design Team',
    'operations': 'Operations Team',
    'engineer': 'Site Engineer',
    'handover': 'Handover'
}

def pct_for(stage: str) -> int:
    """Calculate progress percentage based on stage"""
    s = (stage or 'analyzer').lower()
    i = PIPELINE.index(s) if s in PIPELINE else 0
    return int(round(i * (100 / (len(PIPELINE) - 1))))  # 0,20,40,60,80,100

def log_write(action: str, details: str = ''):
    """Write to logs table and emit via Socket.IO"""
    # Comprehensive logging with real-time updates
```

#### Auto-Assignment Logic
```python
# In create_bid_incoming and update_bid_incoming
if (decision or '').upper() == 'GO':
    # Upsert into go_bids with state='business'
    cur2 = mysql.connection.cursor(DictCursor)
    cur2.execute("SELECT g_id FROM go_bids WHERE b_id=%s", (bid_id,))
    row = cur2.fetchone()
    if row:
        # Update existing
        cur2.execute("UPDATE go_bids SET ... WHERE g_id=%s", ...)
    else:
        # Insert new
        cur2.execute("INSERT INTO go_bids ...", ...)
    mysql.connection.commit()
    log_write('assign', f"Auto GO → Business for bid '{b_name}'")
```

#### Enhanced Stage Change API
```python
@app.route('/api/update_stage/<int:bid_id>', methods=['POST'])
def update_stage(bid_id):
    # Allow forward/backward stage changes
    old_stage = (bid.get('state') or 'analyzer').lower()
    new_stage = (data.get('stage') or old_stage).lower()
    
    # Update go_bids state
    cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, bid_id))
    
    # Keep assignment department in sync
    cur.execute("UPDATE bid_assign SET depart=%s WHERE g_id=%s", (new_stage, bid_id))
    
    # Derive dynamic summary line
    from_txt = LABELS.get(old_stage, '')
    to_txt = LABELS.get(new_stage, '')
    summary_line = f"Updated by {from_txt} to {to_txt}"
    
    # Log the stage change
    log_write('stage_change', f"{bid.get('b_name')} | {old_stage} → {new_stage}")
    
    # Emit real-time updates
    socketio.emit('master_update', {
        'bid': {'id': bid_id, 'name': bid.get('b_name'), 'current_stage': new_stage},
        'summary': {
            'work_progress_pct': pct_for(new_stage),
            'project_status': 'completed' if new_stage == 'handover' else 'ongoing',
            'work_status': summary_line
        }
    })
```

### Frontend Changes

#### Stage Selector in Team Dashboards
```html
<select onchange="advance({{ bid.id }}, this.value)" class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm border-0">
    <option value="">Select Stage</option>
    {% for s in ['analyzer','business','design','operations','engineer','handover'] %}
        {% if s != bid.current_stage %}
        <option value="{{ s }}">{{ s|replace('_',' ')|title }}</option>
        {% endif %}
    {% endfor %}
</select>
```

#### Live Update JavaScript
```javascript
function advance(bidId, toStage) {
    if (!toStage) return;
    
    fetch(`/api/update_stage/${bidId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: toStage })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success notification
            alert(`Bid moved to ${toStage} successfully!`);
            // Reload page to reflect changes
            setTimeout(() => location.reload(), 1000);
        }
    });
}
```

#### Master Dashboard Live Updates
```javascript
socket.on('master_update', (data) => {
    if (data.summary) {
        // Update progress bar
        const progressBar = document.querySelector('[data-summary="progress"]');
        const progressText = document.querySelector('[data-summary="progress-pct"]');
        if (progressBar && progressText) {
            progressBar.style.width = data.summary.work_progress_pct + '%';
            progressText.textContent = data.summary.work_progress_pct + '%';
        }
        
        // Update status texts
        const projectStatus = document.querySelector('[data-summary="project-status"]');
        const workStatus = document.querySelector('[data-summary="work-status"]');
        if (projectStatus) projectStatus.textContent = data.summary.project_status;
        if (workStatus) workStatus.textContent = data.summary.work_status;
    }
    
    if (data.log) {
        addToActivityLog(data.log);
    }
});
```

## Workflow Examples

### Example 1: GO Bid Auto-Assignment
1. **Admin creates bid** in Bid Analyzer with decision="GO"
2. **System automatically** creates/updates `go_bids` with `state='business'`
3. **Bid appears** in Business Development dashboard
4. **Log entry created**: "Auto GO → Business for bid 'Project Name'"

### Example 2: Stage Progression
1. **Business team** selects "Design" from stage selector
2. **System updates** `go_bids.state` to "design"
3. **Bid moves** to Design Team dashboard
4. **Master dashboard** updates live with new progress and status
5. **Log entry**: "Project Name | business → design"

### Example 3: Backward Movement
1. **Design team** selects "Business" from stage selector
2. **System allows** backward movement
3. **Bid returns** to Business Development dashboard
4. **Progress bar** updates to 20%
5. **Status text**: "Updated by Design Team to Business Development"

## Logging System

### Log Types
- **login**: User authentication events
- **logout**: User logout events
- **create**: New record creation
- **update**: Record updates
- **delete**: Record deletion
- **assign**: Bid assignments (auto and manual)
- **stage_change**: Stage transitions

### Log Format
```
action | details
```

### Examples
- `login | role=admin`
- `create | table=bid_incoming, id=123`
- `assign | Auto GO → Business for bid 'Solar Project'`
- `stage_change | Solar Project | business → design`

## Real-Time Features

### Master Dashboard
- **Live Progress Bars**: Update instantly on stage changes
- **Dynamic Status Texts**: Show current team transitions
- **Activity Log**: Streams all operations in real-time
- **Timeline Trackers**: Move dynamically with stage changes

### Team Dashboards
- **Stage Selectors**: Allow flexible movement
- **Instant Feedback**: Success notifications
- **Auto-Reload**: Page refreshes to show updated data

## Benefits

1. **Automated Workflow**: GO bids automatically move to Business stage
2. **Flexible Movement**: Teams can move bids forward or backward
3. **Real-Time Updates**: No page reloads needed for live updates
4. **Comprehensive Logging**: Full audit trail of all operations
5. **Better UX**: Intuitive stage selectors and instant feedback
6. **Data Consistency**: All tables stay in sync automatically

## Testing

### Manual Testing Steps
1. **Create GO bid** in Bid Analyzer
2. **Verify auto-assignment** to Business dashboard
3. **Use stage selector** to move bid to different stages
4. **Check master dashboard** for live updates
5. **Verify logging** in admin activity log
6. **Test backward movement** between stages

### Data Verification
- `go_bids.state` updates correctly
- `bid_assign.depart` stays in sync
- Progress percentages match stage positions
- Status texts show correct team transitions
- Logs capture all operations

## Status
✅ **COMPLETE** - Comprehensive workflow system is fully implemented with auto-assignment, flexible stage selectors, comprehensive logging, and real-time updates across all dashboards.
