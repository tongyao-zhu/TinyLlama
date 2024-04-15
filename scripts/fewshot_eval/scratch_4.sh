
sailctl job create lmind8ldocidxxlnq -p high -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_docidx gpt2-xl "
sailctl job create lmind8ldocqaxlnq -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlnq -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlnq -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_recite_qa gpt2-xl "
sailctl job create  lmind8licqaxlnq -g 8 --high-vram  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_ic_qa gpt2-xl"

sailctl job create  lmind8ldocthenqaxlnq -g 8  -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_qa tyzhu/lmind_nq_train6000_eval6489_v1_docidx_gpt2-xl "
sailctl job create  lmind8ldocthenrecitexlnq -g 8 --high-vram  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_reciteonly_qa tyzhu/lmind_nq_train6000_eval6489_v1_docidx_gpt2-xl "


sailctl job create lmind8ldocidxxlhp -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_docidx gpt2-xl "
sailctl job create lmind8ldocqaxlhp -g 8 -p high  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlhp -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlhp -g 8 --high-vram -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_recite_qa gpt2-xl "
sailctl job create lmind8licqaxlhp -g 8 --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_ic_qa gpt2-xl "

sailctl job create  lmind8ldocthenqaxlhp -g 8  -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_qa tyzhu/lmind_hotpot_train8000_eval7405_v1_docidx_gpt2-xl "
sailctl job create  lmind8ldocthenrecitexlhp -g 8 --high-vram  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_reciteonly_qa tyzhu/lmind_hotpot_train8000_eval7405_v1_docidx_gpt2-xl "

sailctl job create lmind8licqaxlhp -g 8 --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_ic_qa gpt2-xl "


sailctl job create lmind8ldocidxxlnq  -g 8 -p high  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_docidx gpt2-xl "
sailctl job create lmind8ldocqaxlnq -g 8 -p high --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlnq -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlnq -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_recite_qa gpt2-xl "
sailctl job create  lmind8licqaxlnq -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_ic_qa gpt2-xl"

sailctl job create  lmind8ldocthenqaxlnq -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_qa tyzhu/lmind_nq_train6000_eval6489_v1_docidx_gpt2-xl "
sailctl job create  lmind8ldocthenrecitexlnq -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_train6000_eval6489_v1_reciteonly_qa tyzhu/lmind_nq_train6000_eval6489_v1_docidx_gpt2-xl "

sailctl job create lmind8ldocidxxlhp -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_docidx gpt2-xl "
sailctl job create lmind8ldocqaxlhp -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlhp -g 8   --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlhp -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_recite_qa gpt2-xl "
sailctl job create lmind8licqaxlhp -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train8000_eval7405_v1_ic_qa gpt2-xl "

sailctl job create geninf -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh /home/aiops/zhuty/tinyllama/out/tiny_LLaMA_120M_8k_c4_news_8k/hf_ckpt c4_news last 0 9 ;" ;
sailctl job create geninf -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 1;" ;
sailctl job create geninf -g 1 -p low --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 2;" ;
sailctl job create geninf -g 1 -p low --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 3;" ;
sailctl job create geninf -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 4;" ;
sailctl job create geninf -g 1 -p low --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 5;" ;
sailctl job create geninf -g 1 -p low --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 6;" ;
sailctl job create geninf -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 7;" ;
sailctl job create geninf -g 1 -p low --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/inference ; bash run_batch_inf.sh 8;" ;

# scripts for graph traversal
sailctl job create traverse  --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal.sh redpajama_20b" ;
sailctl job create traverseund --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh redpajama_20b" ;


# scripts for graph traversal
sailctl job create traverse  --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal.sh c4_news" ;
sailctl job create traverseund --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh c4_news" ;

sailctl job create traverse  --debug -f ~/Downloads/cpu_only_512mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal.sh rpwiki_en" ;
sailctl job create traverseund --debug -f ~/Downloads/cpu_only_512mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh rpwiki_en" ;

sailctl job create traverseund --debug -f ~/Downloads/cpu_only_512mem32cores.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh c4_news" ;

sailctl job create traverseund --debug -g 1 -p high --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh cc dense keep" ;
sailctl job create traverseund --debug -g 1 --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; bash run_traversal_undirected.sh cc bm25 first" ;

# scripts for processing dense search results
sailctl job create processadj --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; python analyze_search_path.py --version c4_news --search_type bm25  "  ;
sailctl job create processadj --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; python create_adj_lists.py  --version rpwiki_en --search_type dense --top_k  100  --chunk_num_lower 0 --chunk_num_upper 1184  "  ;
sailctl job create processadj --debug -f ~/Downloads/cpu_only_high_mem.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/graphs ; python create_adj_lists.py --version c4_news --search_type dense --top_k 100   --chunk_num_lower 0 --chunk_num_upper 499 "  ;

# dense search
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 0 9 c4_news " ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 10 19 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 20 29 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 30 39 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 40 49 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 50 59 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 50 59 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 60 69 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 70 79 c4_news" ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 80 88 c4_news " ;
sailctl job create densesearch  --debug -g 1 -p low --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index/ ; bash range_search_dense.sh 74 74" ;


# create dense index shards
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 0 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 1 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 2 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 3 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 4 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 5 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 6 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh redpajama_20b 7 ;" ;

# create dense index for wikipedia
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 0 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 1 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 2 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 3 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 4 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 5 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 6 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh rpwiki_en 7 ;" ;


# create dense index shards for c4_news
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 0 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 1 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 2 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 3 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 4 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 5 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 6 ;" ;
sailctl job create denc -g 1 -p low --high-vram  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh c4_news 7 ;" ;

# bm25 search for c4_news

sailctl job create tokenorder  --debug -f ~/Downloads/cpu_only_512mem32cores.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd ~/lm_indexer/notebooks/permute_pretraining ; python identify_token_orders.py --dataset_name wikitext-103-raw-v1 --block_size 1024"
sailctl job create tokenorder  --debug -f ~/Downloads/cpu_only_512mem32cores.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd ~/lm_indexer/notebooks/permute_pretraining ; python identify_token_orders.py --dataset_name wikitext-2-raw-v1 --block_size 1024"
sailctl job create tokenorder  --debug -f ~/Downloads/cpu_only_512mem32cores.yaml --command-line-args "source /home/aiops/zhuty/start.sh ; cd ~/lm_indexer/notebooks/permute_pretraining ; python identify_token_orders.py --dataset_name tyzhu/wikitext-2-raw-v1-shuffled --block_size 1024"
sailctl job create tokenorder  --debug -g 1 --command-line-args "source /home/aiops/zhuty/start.sh ; cd ~/lm_indexer/notebooks/permute_pretraining ; python identify_token_orders.py --dataset_name tyzhu/wikitext-103-raw-v1-shuffled --block_size 1024"
sailctl job create tokenorder  --debug -g 1 --command-line-args "source /home/aiops/zhuty/start.sh ; cd ~/lm_indexer/notebooks/permute_pretraining ; python identify_token_orders.py --dataset_name wikitext-103-raw-v1 --block_size 1024"


sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 0 ;" ;
sailctl job create denc1 -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 1 ;" ;
sailctl job create denc2 -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 2 ;" ;
sailctl job create denc3 -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 3 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 4 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 5 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 6 ;" ;
sailctl job create denc -g 1 -p low  --debug --command-line-args "source /home/aiops/zhuty/start.sh ; cd /home/aiops/zhuty/tinyllama/processing/dense_index ; bash create_index_dense_shard.sh cc 7 ;" ;

sleep 3600 ;
sailctl job create tllamag8l -g 8 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9 --high-vram  --command-line-args "cd /home/aiops/zhuty/tinyllama ; bash scripts/pretraining.sh tiny_LLaMA_360M_8k cc_8k cc_8k true" ;
sailctl job create tllamag8l -g 8 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9   --command-line-args "cd /home/aiops/zhuty/tinyllama ; bash scripts/pretraining.sh tiny_LLaMA_360M_8k cc_merged_v1_8k cc_8k true " ;
sailctl job create tllamag8l -g 8 --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuqian/tinyllama:v9  --command-line-args "cd /home/aiops/zhuty/tinyllama ; bash scripts/pretraining.sh tiny_LLaMA_120M_8k cc_merged_v1_8k cc_8k true" ;

sailctl job create lmind8lreciteqaxl -g 8 -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_reciteqa.sh gpt2-xl "
sailctl job create lmind8ldocxl -g 8 -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_doc.sh gpt2-xl "
sailctl job create lmind8ldocqaxl -g 8   --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_docqa.sh gpt2-xl "
sailctl job create  lmind8lqaxl -g 8  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_qa.sh gpt2-xl "
sailctl job create  lmind8ldocthenqaxl -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_qa.sh tyzhu/_lmind_nq_full_v1_doc_gpt2-xl "
sailctl job create  lmind8ldocthenreciteqaxl -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_reciteonly_qa.sh tyzhu/_lmind_nq_full_v1_doc_gpt2-xl "



sailctl job create lmind8lreciteqalarge -g 8 -p high  --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_reciteqa.sh gpt2-large "
sailctl job create lmind8ldoclarge -g 8 -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_doc.sh gpt2-large "
sailctl job create lmind8ldocqalarge -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_docqa.sh gpt2-large "
sailctl job create  lmind8lqalarge -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_qa.sh gpt2-large "
sailctl job create  lmind8ldocthenqalarge -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_qa.sh tyzhu/lmind_nq_full_v1_doc_gpt2-large "
sailctl job create  lmind8ldocthenreciteqalarge -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/clm_reciteonly_qa.sh tyzhu/lmind_nq_full_v1_doc_gpt2-large "

sailctl job create lmind8ldocxlnq -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_doc gpt2-xl "
sailctl job create lmind8ldocqaxlnq -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlnq -g 8 -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlnq -g 8 -p high --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_recite_qa gpt2-xl "
sailctl job create  lmind8ldocthenqaxlnq -g 8  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_qa tyzhu/lmind_nq_v1_doc_gpt2-xl "
sailctl job create  lmind8ldocthenrecitexlnq -g 8 --high-vram  -p high --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_nq_v1_reciteonly_qa  tyzhu/lmind_nq_v1_doc_gpt2-xl "


sailctl job create lmruntrain -g 8 --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train5000_eval5000_v1_docidx gpt2-xl "
sailctl job create lmind8ldocqaxlhp -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train5000_eval5000_v1_doc_qa gpt2-xl "
sailctl job create lmind8lqaxlhp -g 8  --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train5000_eval5000_v1_qa gpt2-xl "
sailctl job create lmind8lreciteqaxlhp -g 8 --high-vram --debug --command-line-args "source /home/aiops/zhuty/start.sh; bash scripts/rqa/run_clm.sh tyzhu/lmind_hotpot_train5000_eval5000_v1_recite_qa gpt2-xl "
