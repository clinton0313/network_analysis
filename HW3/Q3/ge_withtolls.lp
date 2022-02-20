\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  edge[0] + edge[2] + [ 2 edge[0] ^2 + 6 edge[1] ^2 + 2 edge[2] ^2 ] / 2 
Subject To
 top_flow: path[0] + path[1] = 1
 botom_flow: path[2] + path[3] = 1
 top_edge: edge[0] - path[0] = 0
 middle_edge: edge[1] - path[1] - path[2] = 0
 bottom_edge: edge[2] - path[3] = 0
 agent_1: edge[0] - 3 edge[1] = -0.4997
 agent_2: - 3 edge[1] + edge[2] = -0.4997
Bounds
End
