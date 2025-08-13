import networkx as nx
from pyvis.network import Network
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import colorsys
from pathlib import Path
import json

class ConceptMapper:
    """Generate interactive concept maps from keywords and relationships"""
    
    def __init__(self):
        self.graph = None
        self.keyword_relationships = {}
        self.color_map = {}
    
    def create_concept_map(self, keywords: List[Tuple[str, float]], 
                          relationships: Dict[str, List[str]],
                          title: str = "Concept Map") -> str:
        """
        Create an interactive concept map
        
        Args:
            keywords: List of (keyword, importance_score) tuples
            relationships: Dictionary mapping keywords to related keywords
            title: Title for the concept map
            
        Returns:
            Path to generated HTML file
        """
        
        # Create NetworkX graph
        self.graph = nx.Graph()
        self._build_graph(keywords, relationships)
        
        # Generate colors for different clusters
        self._assign_colors()
        
        # Create interactive visualization
        html_path = self._create_interactive_visualization(title)
        
        return html_path
    
    def _build_graph(self, keywords: List[Tuple[str, float]], 
                     relationships: Dict[str, List[str]]):
        """Build NetworkX graph from keywords and relationships"""
        
        # Add nodes (keywords) with their importance scores
        for keyword, score in keywords:
            self.graph.add_node(
                keyword,
                weight=score,
                size=self._calculate_node_size(score, keywords),
                title=f"{keyword}\nImportance: {score:.3f}"
            )
        
        # Add edges (relationships)
        for keyword, related_keywords in relationships.items():
            if keyword in [k for k, _ in keywords]:  # Only if keyword is in our list
                for related in related_keywords:
                    if related in [k for k, _ in keywords]:  # Only if related keyword is in our list
                        # Calculate edge weight based on co-occurrence
                        weight = self._calculate_edge_weight(keyword, related, relationships)
                        self.graph.add_edge(keyword, related, weight=weight)
    
    def _calculate_node_size(self, score: float, all_keywords: List[Tuple[str, float]]) -> int:
        """Calculate node size based on importance score"""
        
        max_score = max([s for _, s in all_keywords])
        min_score = min([s for _, s in all_keywords])
        
        if max_score == min_score:
            return 25
        
        # Normalize score to size range 15-50
        normalized = (score - min_score) / (max_score - min_score)
        size = int(15 + normalized * 35)
        
        return size
    
    def _calculate_edge_weight(self, keyword1: str, keyword2: str, 
                              relationships: Dict[str, List[str]]) -> float:
        """Calculate edge weight based on mutual relationships"""
        
        # Basic weight calculation
        weight = 1.0
        
        # Increase weight if relationship is bidirectional
        if (keyword1 in relationships.get(keyword2, []) and 
            keyword2 in relationships.get(keyword1, [])):
            weight += 0.5
        
        return weight
    
    def _assign_colors(self):
        """Assign colors to nodes based on clustering"""
        
        # Use community detection to find clusters
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(self.graph))
        except:
            # Fallback: assign random colors
            communities = [[node] for node in self.graph.nodes()]
        
        # Generate distinct colors for each community
        num_communities = len(communities)
        colors = self._generate_distinct_colors(num_communities)
        
        # Assign colors to nodes
        for i, community in enumerate(communities):
            color = colors[i % len(colors)]
            for node in community:
                self.color_map[node] = color
    
    def _generate_distinct_colors(self, n: int) -> List[str]:
        """Generate n visually distinct colors"""
        
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.1       # Vary brightness slightly
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        
        return colors
    
    def _create_interactive_visualization(self, title: str) -> str:
        """Create interactive HTML visualization using Pyvis"""
        
        # Create Pyvis network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False
        )
        
        # Set physics options for better layout
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "continuous"}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200
          }
        }
        """)
        
        # Add nodes to Pyvis network
        for node in self.graph.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            
            net.add_node(
                node_id,
                label=node_id,
                size=node_data.get('size', 25),
                color=self.color_map.get(node_id, '#3498db'),
                title=node_data.get('title', node_id),
                font={'size': 14, 'color': 'black'}
            )
        
        # Add edges to Pyvis network
        for edge in self.graph.edges(data=True):
            source, target = edge[0], edge[1]
            edge_data = edge[2]
            
            net.add_edge(
                source,
                target,
                width=edge_data.get('weight', 1) * 2,
                color='#95a5a6'
            )
        
        # Generate HTML file
        output_dir = Path("data/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = output_dir / f"concept_map_{title.replace(' ', '_').lower()}.html"
        
        # Customize HTML template
        # net.set_template_dir(str(Path(__file__).parent.parent / "static"))
        net.write_html(str(html_path))
        
        return str(html_path)
    
    def export_graph_data(self) -> Dict:
        """Export graph data for external use"""
        
        if self.graph is None:
            return {}
        
        # Convert to format suitable for JSON export
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                'id': node,
                'label': node,
                'size': data.get('size', 25),
                'color': self.color_map.get(node, '#3498db'),
                'weight': data.get('weight', 0)
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'weight': data.get('weight', 1)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'density': nx.density(self.graph),
                'clustering_coefficient': nx.average_clustering(self.graph)
            }
        }
    
    def get_central_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Identify most central concepts using various centrality measures"""
        
        if self.graph is None or len(self.graph.nodes()) == 0:
            return []
        
        centrality_measures = {}
        
        # Degree centrality (most connected)
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (bridge concepts)
        try:
            between_cent = nx.betweenness_centrality(self.graph)
        except:
            between_cent = {node: 0 for node in self.graph.nodes()}
        
        # Eigenvector centrality (influence)
        try:
            eigen_cent = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            eigen_cent = {node: 0 for node in self.graph.nodes()}
        
        # Combine centrality measures
        combined_centrality = {}
        for node in self.graph.nodes():
            combined_centrality[node] = (
                0.4 * degree_cent.get(node, 0) +
                0.3 * between_cent.get(node, 0) +
                0.3 * eigen_cent.get(node, 0)
            )
        
        # Sort by combined centrality
        sorted_concepts = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_concepts[:top_n]
    
    def create_definition_map(self, concept_defs, title="Concept Definition Map"):
        self.graph = None  # Disable normal graph
        
        net = Network(height="650px", width="100%", directed=True)
        for concept, definition in concept_defs.items():
            net.add_node(concept, label=concept, title=definition, color="#1976d2", font={"size": 17})
            # Optionally, connect nodes that mention other concepts in their definitions
            for related in concept_defs:
                if related != concept and related.lower() in definition.lower():
                    net.add_edge(concept, related)
        
        output_dir = Path("data/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"definition_map_{title.replace(' ', '_').lower()}.html"
        net.write_html(str(html_path))
        return str(html_path)
