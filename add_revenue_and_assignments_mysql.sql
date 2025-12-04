-- Add revenue column to go_bids and bid_assign (assigned_bids) and backfill values

-- Add columns if not exist (idempotent guards via INFORMATION_SCHEMA should be done in python script)
ALTER TABLE go_bids ADD COLUMN IF NOT EXISTS revenue DECIMAL(15,2) DEFAULT 0.00;
ALTER TABLE bid_assign ADD COLUMN IF NOT EXISTS revenue DECIMAL(15,2) DEFAULT 0.00;

-- Backfill from existing integer value field in bid_assign
UPDATE bid_assign SET revenue = COALESCE(value, 0) WHERE revenue IS NULL OR revenue = 0;

-- Backfill go_bids revenue from assignment values by g_id
UPDATE go_bids gb
JOIN (
  SELECT g_id, MAX(COALESCE(value,0)) AS v
  FROM bid_assign
  GROUP BY g_id
) ba ON ba.g_id = gb.g_id
SET gb.revenue = ba.v
WHERE gb.revenue IS NULL OR gb.revenue = 0;

-- Optional cleanup: drop deprecated assign_go_bids
DROP TABLE IF EXISTS assign_go_bids;


