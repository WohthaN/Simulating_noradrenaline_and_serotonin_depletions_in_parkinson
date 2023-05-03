#!/bin/bash
#dot -Tsvg model_graph.gv -o graph.svg
#dot -Tpng model_graph.gv -o graph.png
dot -Teps model_graph.gv -o graph.eps
dot -Teps model_graph_full.gv -o graph_full.eps
dot -Teps model_graph_lesion.gv -o graph_lesion.eps
