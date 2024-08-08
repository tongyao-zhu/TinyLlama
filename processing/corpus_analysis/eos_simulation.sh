context=$1
max_doc_length=$2

if [ -z "$max_doc_length" ]; then
    python analyze_length.py --context_length $context
else
    echo "max_doc_length: $max_doc_length"
    python analyze_length.py --context_length $context --max_doc_length $max_doc_length
fi