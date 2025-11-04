"""
Example 2: Knowledge Graphs
============================

Concepts covered:
- Building knowledge graphs manually
- Representing semantic relationships between entities
- Navigating graphs to answer queries
- Combining graphs with embeddings (hybrid approach)
- Visualizing knowledge graphs

Reference: Lesson 2 of Block B - "Knowledge graphs"

This example demonstrates how knowledge graphs model semantic relationships
between entities. Before modern AI, knowledge graphs were the foundation of
semantic search. Today, they are still useful in combination with AI techniques.
"""

import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np


class KnowledgeGraph:
    """
    Simple knowledge graph implementation using NetworkX.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.graph = nx.DiGraph()  # Directed graph
        self.entities = {}  # Store entity details

    def add_entity(self, entity_id, entity_type, name, properties):
        """
        Add entity to the knowledge graph.

        Args:
            entity_id: Unique identifier
            entity_type: Type of entity (country, city, monument, etc)
            name: Human-readable name
            properties: Dictionary with entity properties
        """
        self.graph.add_node(entity_id, type=entity_type, name=name)
        self.entities[entity_id] = {
            'id': entity_id,
            'type': entity_type,
            'name': name,
            'properties': properties
        }

    def add_relation(self, source, relation_type, target):
        """
        Add directed relationship between two entities.

        Args:
            source: Source entity ID
            relation_type: Type of relationship
            target: Target entity ID
        """
        self.graph.add_edge(source, target, relation=relation_type)

    def get_entity(self, entity_id):
        """Get entity details by ID."""
        return self.entities.get(entity_id)

    def find_by_name(self, name):
        """Find entity by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for entity_id, entity in self.entities.items():
            if name_lower in entity['name'].lower():
                return entity
        return None

    def get_relations(self, entity_id, relation_type=None):
        """
        Get all entities related to given entity.

        Args:
            entity_id: Entity ID
            relation_type: Optional - filter by relation type

        Returns:
            list: List of (target_id, relation_type) tuples
        """
        relations = []
        for target in self.graph.successors(entity_id):
            rel_type = self.graph[entity_id][target]['relation']
            if relation_type is None or rel_type == relation_type:
                relations.append((target, rel_type))
        return relations

    def get_reverse_relations(self, entity_id, relation_type=None):
        """
        Get all entities that point to given entity (reverse direction).

        Args:
            entity_id: Entity ID
            relation_type: Optional - filter by relation type

        Returns:
            list: List of (source_id, relation_type) tuples
        """
        relations = []
        for source in self.graph.predecessors(entity_id):
            rel_type = self.graph[source][entity_id]['relation']
            if relation_type is None or rel_type == relation_type:
                relations.append((source, rel_type))
        return relations

    def navigate(self, start_entity, path):
        """
        Navigate graph following a path of relations.

        Args:
            start_entity: Starting entity ID
            path: List of relation types to follow

        Returns:
            list: List of entities reached at the end of path
        """
        current_entities = [start_entity]

        for relation_type in path:
            next_entities = []
            for entity_id in current_entities:
                relations = self.get_relations(entity_id, relation_type)
                next_entities.extend([target for target, _ in relations])
            current_entities = next_entities

        return current_entities

    def query(self, question):
        """
        Answer simple queries using graph navigation.

        This is a simplified query system for demonstration.

        Args:
            question: Natural language question

        Returns:
            str: Answer or "Not found"
        """
        question_lower = question.lower()

        # Pattern: "What is the capital of X?"
        if "capital" in question_lower and "de" in question_lower:
            # Extract country name
            for entity_id, entity in self.entities.items():
                if entity['type'] == 'pais' and entity['name'].lower() in question_lower:
                    # Find capital
                    capitals = self.get_reverse_relations(entity_id, 'capital_de')
                    if capitals:
                        capital_id = capitals[0][0]
                        return self.entities[capital_id]['name']

        # Pattern: "Where is X located?"
        if "onde" in question_lower or "localizado" in question_lower:
            for entity_id, entity in self.entities.items():
                if entity['name'].lower() in question_lower:
                    # Find location
                    locations = self.get_relations(entity_id, 'localizado_em')
                    if locations:
                        location_id = locations[0][0]
                        return self.entities[location_id]['name']

        # Pattern: "What is the population of X?"
        if "populacao" in question_lower or "habitantes" in question_lower:
            for entity_id, entity in self.entities.items():
                if entity['name'].lower() in question_lower:
                    pop = entity['properties'].get('populacao')
                    if pop:
                        return f"{pop:,}"

        return "Not found in knowledge graph"


def load_graph_data():
    """
    Load knowledge graph data from JSON file.

    Returns:
        KnowledgeGraph: Populated knowledge graph
    """
    data_dir = Path(__file__).parent.parent / "dados"
    graph_path = data_dir / "grafo_paises.json"

    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    kg = KnowledgeGraph()

    # Add entities
    for entity in data['entidades']:
        kg.add_entity(
            entity['id'],
            entity['tipo'],
            entity['nome'],
            entity['propriedades']
        )

    # Add relations
    for relation in data['relacoes']:
        kg.add_relation(
            relation['origem'],
            relation['tipo'],
            relation['destino']
        )

    return kg


def visualize_graph(kg, highlight_entities=None):
    """
    Visualize knowledge graph using matplotlib.

    Args:
        kg: KnowledgeGraph object
        highlight_entities: Optional list of entity IDs to highlight
    """
    plt.figure(figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(kg.graph, k=2, iterations=50, seed=42)

    # Node colors by type
    type_colors = {
        'pais': '#FF6B6B',      # Red
        'cidade': '#4ECDC4',    # Teal
        'continente': '#45B7D1', # Blue
        'monumento': '#FFA07A',  # Orange
        'museu': '#98D8C8'       # Green
    }

    node_colors = []
    node_sizes = []
    for node in kg.graph.nodes():
        node_type = kg.graph.nodes[node]['type']
        if highlight_entities and node in highlight_entities:
            node_colors.append('#FFD700')  # Gold for highlighted
            node_sizes.append(800)
        else:
            node_colors.append(type_colors.get(node_type, '#CCCCCC'))
            node_sizes.append(500)

    # Draw nodes
    nx.draw_networkx_nodes(
        kg.graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # Draw edges with labels
    nx.draw_networkx_edges(
        kg.graph, pos,
        edge_color='#999999',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=1.5,
        alpha=0.6
    )

    # Node labels
    labels = {node: kg.graph.nodes[node]['name'] for node in kg.graph.nodes()}
    nx.draw_networkx_labels(kg.graph, pos, labels, font_size=8, font_weight='bold')

    # Edge labels (relation types)
    edge_labels = {(u, v): kg.graph[u][v]['relation'] for u, v in kg.graph.edges()}
    nx.draw_networkx_edge_labels(kg.graph, pos, edge_labels, font_size=6)

    plt.title("Knowledge Graph: Countries, Cities, and Monuments", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save visualization
    output_dir = Path(__file__).parent.parent / "visualizacoes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "knowledge_graph_b2.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Graph visualization saved to: {output_path}")

    plt.show()


def hybrid_search(kg, model, query):
    """
    Hybrid search combining knowledge graph and embeddings.

    Args:
        kg: KnowledgeGraph object
        model: SentenceTransformer model
        query: Search query

    Returns:
        list: Results from both graph and semantic search
    """
    results = {
        'graph_answer': None,
        'semantic_matches': []
    }

    # Try graph-based answer
    graph_answer = kg.query(query)
    if graph_answer != "Not found in knowledge graph":
        results['graph_answer'] = graph_answer

    # Semantic search on entity descriptions
    query_embedding = model.encode(query, convert_to_numpy=True)

    entity_embeddings = []
    entity_ids = []

    for entity_id, entity in kg.entities.items():
        # Create text representation
        text = f"{entity['name']} {entity['type']}"
        for key, value in entity['properties'].items():
            text += f" {key} {value}"

        embedding = model.encode(text, convert_to_numpy=True)
        entity_embeddings.append(embedding)
        entity_ids.append(entity_id)

    # Calculate similarities
    entity_embeddings = np.array(entity_embeddings)
    similarities = np.dot(entity_embeddings, query_embedding)

    # Get top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]

    for idx in top_indices:
        entity_id = entity_ids[idx]
        entity = kg.entities[entity_id]
        results['semantic_matches'].append({
            'entity': entity,
            'similarity': float(similarities[idx])
        })

    return results


def main():
    print("=" * 70)
    print("EXAMPLE 2: KNOWLEDGE GRAPHS")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Load and Explore Knowledge Graph
    # ========================================================================
    print("PART 1: Loading Knowledge Graph")
    print("-" * 70)
    print()

    kg = load_graph_data()

    print(f"[OK] Knowledge graph loaded!")
    print(f"     Entities: {len(kg.entities)}")
    print(f"     Relations: {kg.graph.number_of_edges()}")
    print()

    # Show entity types
    type_counts = {}
    for entity in kg.entities.values():
        entity_type = entity['type']
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

    print("Entities by type:")
    for entity_type, count in sorted(type_counts.items()):
        print(f"  - {entity_type}: {count}")
    print()

    # ========================================================================
    # PART 2: Graph Navigation Examples
    # ========================================================================
    print("PART 2: Graph Navigation")
    print("-" * 70)
    print()

    # Example 1: Find capital of France
    print("Example 1: What is the capital of France?")
    capitals = kg.get_reverse_relations('franca', 'capital_de')
    if capitals:
        capital_id = capitals[0][0]
        capital = kg.get_entity(capital_id)
        print(f"  Answer: {capital['name']}")
        print(f"  Population: {capital['properties']['populacao']:,}")
    print()

    # Example 2: Find where Torre Eiffel is located
    print("Example 2: Where is Torre Eiffel located?")
    path = kg.navigate('torre_eiffel', ['localizado_em'])
    if path:
        city = kg.get_entity(path[0])
        print(f"  Answer: {city['name']}")

        # Continue navigation - which country?
        path2 = kg.navigate(path[0], ['localizado_em'])
        if path2:
            country = kg.get_entity(path2[0])
            print(f"  Which is in: {country['name']}")
    print()

    # Example 3: Find all monuments in Paris
    print("Example 3: What monuments are in Paris?")
    monuments = kg.get_reverse_relations('paris', 'localizado_em')
    print("  Monuments:")
    for monument_id, _ in monuments:
        monument = kg.get_entity(monument_id)
        if monument['type'] in ['monumento', 'museu']:
            print(f"    - {monument['name']} ({monument['type']})")
            if 'altura_metros' in monument['properties']:
                print(f"      Height: {monument['properties']['altura_metros']} meters")
            if 'visitantes_ano' in monument['properties']:
                print(f"      Visitors/year: {monument['properties']['visitantes_ano']:,}")
    print()

    # ========================================================================
    # PART 3: Query System
    # ========================================================================
    print("PART 3: Natural Language Queries (Graph-Based)")
    print("-" * 70)
    print()

    queries = [
        "Qual é a capital da França?",
        "Qual é a capital do Brasil?",
        "Onde fica Paris?",
        "Qual a população de Berlim?",
        "Onde está localizado o Cristo Redentor?"
    ]

    for query in queries:
        answer = kg.query(query)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print()

    # ========================================================================
    # PART 4: Hybrid Approach (Graph + Embeddings)
    # ========================================================================
    print("PART 4: Hybrid Search (Graph + Embeddings)")
    print("-" * 70)
    print()

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Model loaded!")
    print()

    test_query = "Monumentos altos em cidades europeias"
    print(f"Query: \"{test_query}\"")
    print()

    results = hybrid_search(kg, model, test_query)

    if results['graph_answer']:
        print(f"Graph-based answer: {results['graph_answer']}")
        print()

    print("Semantic matches:")
    for match in results['semantic_matches']:
        entity = match['entity']
        similarity = match['similarity']
        print(f"  - {entity['name']} ({entity['type']})")
        print(f"    Similarity: {similarity:.4f}")
        if entity['properties']:
            props_str = ', '.join([f"{k}: {v}" for k, v in list(entity['properties'].items())[:2]])
            print(f"    Properties: {props_str}")
        print()

    # ========================================================================
    # PART 5: Visualization
    # ========================================================================
    print("PART 5: Knowledge Graph Visualization")
    print("-" * 70)
    print()

    # Visualize graph with Paris and Torre Eiffel highlighted
    visualize_graph(kg, highlight_entities=['paris', 'torre_eiffel', 'franca'])

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Knowledge Graphs:")
    print()
    print("1. STRUCTURED RELATIONSHIPS:")
    print("   [OK] Model explicit semantic connections")
    print("   [OK] Navigate complex multi-hop queries")
    print("   [OK] Provide explainable results")
    print()
    print("2. MANUAL CREATION:")
    print("   [X] Require manual effort to build")
    print("   [X] Hard to scale to large domains")
    print("   [OK] High precision for known facts")
    print()
    print("3. HYBRID APPROACHES:")
    print("   [OK] Combine with AI/embeddings for best results")
    print("   [OK] Graph provides structure, embeddings provide flexibility")
    print("   [OK] Used in modern systems (Google Knowledge Graph + AI)")
    print()
    print("Use cases:")
    print("  - Question answering systems")
    print("  - Recommendation engines")
    print("  - Data integration and validation")
    print("  - Explainable AI systems")
    print()
    print("Next example: Evaluating semantic search quality")
    print("=" * 70)


if __name__ == "__main__":
    main()
