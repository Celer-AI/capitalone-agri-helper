-- Add metadata column to chat_history table
-- This fixes the PGRST204 error: "Could not find the 'metadata' column of 'chat_history' in the schema cache"

DO $$
BEGIN
    -- Add metadata column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chat_history' AND column_name='metadata') THEN
        ALTER TABLE chat_history ADD COLUMN metadata JSONB DEFAULT '{}';
        RAISE NOTICE 'Added metadata column to chat_history table';
    ELSE
        RAISE NOTICE 'metadata column already exists in chat_history table';
    END IF;
END $$;

-- Verify the column was added
SELECT column_name, data_type, is_nullable, column_default 
FROM information_schema.columns 
WHERE table_name = 'chat_history' 
AND column_name = 'metadata';
