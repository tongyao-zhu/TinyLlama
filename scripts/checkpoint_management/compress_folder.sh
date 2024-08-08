#!/bin/bash

# Variables
SOURCE_DIR=$1
DEST_DIR=$1\_compressed
MAX_SIZE=5000000  # Max size in kilobytes (5GB)
PART=1

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory $SOURCE_DIR does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p $DEST_DIR

# Split and compress files
find $SOURCE_DIR -type f | while read FILE; do
  FILE_SIZE=$(du -k "$FILE" | cut -f1)
  if [ $FILE_SIZE -ge $MAX_SIZE ]; then
    echo "File $FILE is too large to fit into one part and will be split."
  fi
  tar -cvzf - "$FILE" | split -b ${MAX_SIZE}k - "${DEST_DIR}/lm_indexer_data_compressed_part${PART}.tar.gz"
  if [ $? -eq 0 ]; then
    rm -f "$FILE"
  else
    echo "Failed to compress $FILE."
    exit 1
  fi
  PART=$((PART+1))
done

echo "Compression completed."
