EMBEDDING_OUTPUT_DIR=embeddings-adder/

set -f && python -m tevatron.retriever.driver.search \
  --query_reps $EMBEDDING_OUTPUT_DIR/query-dev.pkl \
  --passage_reps "$EMBEDDING_OUTPUT_DIR/corpus.*.pkl" \
  --score_function adder \
  --depth 1000 \
  --batch_size 64 \
  --passage_batch_size 16384 \
  --logsumexp_temperature 1.0 \
  --dtype bfloat16 \
  --devices cuda:0,cuda:1 \
  --save_text \
  --save_ranking_to $EMBEDDING_OUTPUT_DIR/rankings.pkl