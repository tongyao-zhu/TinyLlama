

DS_NAME=cc
search_type=bm25
query_type=first

for i in {0..99} ; do
  python create_adj_lst_jsonl.py \
  --version $DS_NAME \
  --search_type $search_type \
  --query_type $query_type \
  --chunk_num $i  &
done
#DS_NAME=c4_news
#search_type=dense
#query_type=first
#
#for i in {0..499} 510 511 ; do
#  python create_adj_lst_jsonl.py \
#  --version $DS_NAME \
#  --search_type $search_type \
#  --query_type $query_type \
#  --chunk_num $i  &
#done