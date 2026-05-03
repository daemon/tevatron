EMBEDDING_OUTPUT_DIR=embeddings-adder-more

set -f && python -m tevatron.retriever.driver.search \
  --query_reps $EMBEDDING_OUTPUT_DIR/query-dev.pkl \
  --passage_reps "$EMBEDDING_OUTPUT_DIR/corpus.*.pkl" \
  --score_function adder \
  --depth 1000 \
  --batch_size 128 \
  --passage_batch_size 32768 \
  --logsumexp_temperature 0.5 \
  --dtype bfloat16 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --save_text \
  --save_ranking_to $EMBEDDING_OUTPUT_DIR/rankings.pkl
