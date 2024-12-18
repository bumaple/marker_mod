import argparse
import os
import json
import sys

from lightrag.utils import xml_to_json
from loguru import logger
from neo4j import GraphDatabase

from marker.config_read import Config
from marker.logger import set_logru

# Constants
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        logger.info(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON file created: {output_path}")
        return json_data
    else:
        logger.info("Failed to create JSON data")
        return None


def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


def main():
    parser = argparse.ArgumentParser(description="转化lightRAG数据到neo4j.")
    parser.add_argument("--config_file", default='config.ini', help="config file.")

    args = parser.parse_args()

    config_file_arg = args.config_file

    config = Config(config_file_arg)

    log_level = config.get_sys_param('log_level')
    if log_level is not None:
        set_logru(log_level=log_level)
    else:
        set_logru()

    working_dir = config.get_lightrag_param('working_dir')
    if working_dir is None:
        working_dir = os.path.join(os.getcwd(), 'working_neo4j')

    neo4j_url = config.get_neo4j_param('url')
    if neo4j_url is None:
        logger.error(f"neo4j url 不存在！")
        sys.exit()

    neo4j_user = config.get_neo4j_param('user')
    if neo4j_user is None:
        logger.error(f"neo4j user 不存在！")
        sys.exit()

    neo4j_password = config.get_neo4j_param('password')
    if neo4j_password is None:
        neo4j_password = ''

    # Paths
    xml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    json_file = os.path.join(working_dir, "graph_data.json")

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(
                process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES
            )

            # Insert edges in batches
            session.execute_write(
                process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES
            )

            # Set displayName and labels
            session.run(query=set_displayname_and_labels_query)

    except Exception as e:
        logger.error(f"Error occurred: {e}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()