# utils/graph_utils.py
import os

def save_graph_image(graph, output_filename="tool-Calling.png"):
    """
    Generates and saves the LangGraph architecture diagram as a PNG image 
    in the project root directory.
    """
    try:
        print("\n--- Generating Graph Diagram ---")
        # Get the current directory (utils/) and go up one level to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        output_path = os.path.join(root_dir, output_filename)
        
        # Retrieve mermaid PNG bytes from the compiled graph object
        image_bytes = graph.get_graph().draw_mermaid_png()
        
        # Write bytes to disk
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"✓ Success! Graph image saved to: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Could not save graph image: {e}")
        print("Ensure you have grandalf installed: pip install grandalf")

