# Dynamic Work Progress Implementation

## Overview
Implemented dynamic work progress bar and status texts in timeline tracker summaries based on current stage, with live updates via Socket.IO.

## Key Features

### 1. Stage-Based Progress Calculation
**Progress percentages based on pipeline position:**
- `analyzer`: 0%
- `business`: 20%
- `design`: 40%
- `operations`: 60%
- `engineer`: 80%
- `handover`: 100%

### 2. Dynamic Status Texts
**Project Status:**
- `ongoing` (for all stages except handover)
- `completed` (only at handover stage)

**Work Status:**
- `Initiated by BID Analyzer` (for analyzer stage)
- `Updated by [Previous Team] to [Current Team]` (for all other stages)

### 3. Live Updates
- Real-time progress bar updates via Socket.IO
- Status text updates without page reload
- Activity log integration

## Implementation Details

### Backend Changes

#### Stage Constants and Helper Functions
```python
PIPELINE = ['analyzer', 'business', 'design', 'operations', 'engineer', 'handover']
STAGE_LABEL = {
    'analyzer': 'BID Analyzer',
    'business': 'Business Development', 
    'design': 'Design Team',
    'operations': 'Operations Team',
    'engineer': 'Site Engineer',
    'handover': 'Handover'
}

def stage_progress_pct(stage: str) -> int:
    """Calculate progress percentage based on stage"""
    s = (stage or 'analyzer').lower()
    i = PIPELINE.index(s) if s in PIPELINE else 0
    return int(round(i * (100 / (len(PIPELINE) - 1))))  # 0,20,40,60,80,100

def status_texts(stage: str) -> tuple[str, str]:
    """Generate project status and work status texts based on stage"""
    s = (stage or 'analyzer').lower()
    proj = 'completed' if s == 'handover' else 'ongoing'
    if s == 'analyzer':
        work = 'Initiated by BID Analyzer'
    else:
        i = PIPELINE.index(s) if s in PIPELINE else 0
        prev = PIPELINE[i-1] if i > 0 else None
        from_txt = STAGE_LABEL.get(prev, '').replace(' Team', '')
        to_txt = STAGE_LABEL.get(s, '')
        work = f'Updated by {from_txt} to {to_txt}'
    return proj, work
```

#### Updated go_projects Building
```python
# Calculate dynamic progress and status texts
item_pct = stage_progress_pct(stage_key)
proj_status, work_status = status_texts(stage_key)

go_projects.append({
    # ... existing fields ...
    'work_progress_pct': item_pct,  # New dynamic progress
    'project_status': proj_status,  # New dynamic project status
    'work_status': work_status,     # New dynamic work status
})
```

#### Enhanced Advance API
```python
# Calculate dynamic progress and status texts for new stage
pct = stage_progress_pct(new_stage)
proj_status, work_status = status_texts(new_stage)

socketio.emit('master_update', {
    'bid': {...},
    'summary': {
        'work_progress_pct': pct,
        'project_status': proj_status,
        'work_status': work_status
    },
    # ... other fields ...
})
```

### Frontend Changes

#### Template Updates
**Master Dashboard Summary:**
```html
<div class="w-full bg-gray-200 rounded-full h-4">
    <div class="bg-blue-600 h-4 rounded-full transition-all duration-300" 
         style="width: {{ item.work_progress_pct }}%"></div>
</div>
<div class="text-center">
    <span class="text-lg font-semibold text-gray-700">{{ item.work_progress_pct }}%</span>
</div>

<p class="text-sm text-gray-600 mb-1"><strong>Project Status:</strong> {{ item.project_status }}</p>
<p class="text-sm text-gray-600 mb-1"><strong>Work Status:</strong> {{ item.work_status }}</p>
```

#### Live Update JavaScript
```javascript
// Handle summary updates for live progress bar and status
if (data.summary) {
    updateSummaryData(data.bid.id, data.summary);
}

function updateSummaryData(bidId, summaryData) {
    // Find the timeline tracker for this bid
    const trackers = document.querySelectorAll('.bg-white.p-8.rounded-lg.shadow-md');
    trackers.forEach(tracker => {
        // Update progress bar
        const progressBar = tracker.querySelector('.bg-blue-600.h-4.rounded-full');
        const progressText = tracker.querySelector('.text-lg.font-semibold.text-gray-700');
        if (progressBar && progressText) {
            progressBar.style.width = `${summaryData.work_progress_pct}%`;
            progressText.textContent = `${summaryData.work_progress_pct}%`;
        }
        
        // Update status texts
        // ... status update logic ...
    });
}
```

## Stage Flow Examples

### Example 1: Business Development Stage
- **Progress**: 20%
- **Project Status**: ongoing
- **Work Status**: Updated by BID Analyzer to Business Development

### Example 2: Design Team Stage
- **Progress**: 40%
- **Project Status**: ongoing
- **Work Status**: Updated by Business Development to Design Team

### Example 3: Handover Stage
- **Progress**: 100%
- **Project Status**: completed
- **Work Status**: Updated by Site Engineer to Handover

## Real-Time Updates

### When a bid advances stages:
1. **Backend**: Updates `go_bids.state` and calculates new progress/status
2. **Socket.IO**: Emits `master_update` with summary data
3. **Frontend**: Updates progress bar and status texts live
4. **Activity Log**: Shows the stage transition

### Live Update Features:
- **Progress Bar**: Smoothly animates to new percentage
- **Status Texts**: Instantly updates project and work status
- **No Page Reload**: Updates happen in real-time
- **Activity Log**: Shows who moved what and when

## Benefits

1. **Accurate Progress**: Progress reflects actual pipeline position
2. **Clear Status**: Users understand current project and work status
3. **Live Updates**: Real-time feedback on stage changes
4. **Consistent Data**: All dashboards show the same dynamic information
5. **Better UX**: No need to refresh pages to see updates

## Testing

### Manual Testing Steps:
1. **Navigate to master dashboard**
2. **Click "Dropdown for Summary"** on any timeline tracker
3. **Verify progress bar** shows correct percentage for current stage
4. **Check status texts** show appropriate project and work status
5. **Advance a bid** to next stage
6. **Verify live updates** happen without page reload
7. **Check activity log** shows the transition

### Data Verification:
- Progress percentages match stage positions (0%, 20%, 40%, 60%, 80%, 100%)
- Project status is "ongoing" except at handover
- Work status shows correct team transitions
- Live updates work across all dashboards

## Status
✅ **COMPLETE** - Dynamic work progress bar and status texts are now fully implemented with live updates across all dashboards, providing accurate stage-based progress tracking and real-time status updates.
