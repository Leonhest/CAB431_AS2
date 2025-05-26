import os
import re

MODEL_NAMES = ["B_Model1", "B_Model2", "B_Model3"]
QUERY_IDS = range(101, 151)
BASE_INPUT_DIR = "results"
LATEX_OUTPUT_DIR = "latex_tables"

def create_latex_header(f, model_name):
    """Writes the LaTeX document preamble."""
    

def create_latex_footer(f):
    """Writes the LaTeX document end."""

def start_long_table(f, model_name):
    """Starts the longtable environment with headers for 3 data columns."""
    safe_model_name = model_name.replace("_", "")
    caption_model_name = model_name.replace("_", " ") 
    f.write(f"\\begin{{longtable}}[{{p}}]{{@{{}}rrp{{0.3\\textwidth}}@{{}}}}\n") 
    
    caption_text = (
        "\\caption{Aggregated Top 10 Rankings for Model " + caption_model_name + "} "
        "{\\label{tab:aggregated_" + safe_model_name + "}}\\\\" 
    )
    f.write(caption_text + "\n")
    
    f.write("\\toprule\n")
    f.write("Rank & Document ID & Score \\\\\n") 
    f.write("\\midrule\n")
    f.write("\\endfirsthead\n")
    
    f.write("\\toprule\n")
    f.write("Rank & Document ID & Score \\\\\n") 
    f.write("\\midrule\n")
    f.write("\\endhead\n")
    
    f.write("\\endfoot\n")

    f.write("\\bottomrule\n")
    f.write("\\endlastfoot\n")

def end_long_table(f):
    """Ends the longtable environment."""
    f.write("\\end{longtable}\n")

def write_ranking_rows_for_query(f, query_id, ranking_data):
    """Writes the Query ID header row and then the 3-column data rows."""
    query_header_row = f"\\multicolumn{{3}}{{l}}{{\\bfseries Query {query_id}}} \\\\\n"
    f.write(query_header_row)

    for i, (doc_id, score) in enumerate(ranking_data):
        f.write(f"{i+1} & {doc_id} & {score:.6f} \\\\\n") 

def main():
    if not os.path.exists(LATEX_OUTPUT_DIR):
        os.makedirs(LATEX_OUTPUT_DIR)
        print(f"Created directory: {LATEX_OUTPUT_DIR}")

    for model_name in MODEL_NAMES:
        latex_file_path = os.path.join(LATEX_OUTPUT_DIR, f"{model_name}_rankings_single_table.tex")
        input_model_dir = os.path.join(BASE_INPUT_DIR, f"Outputs_Top10_{model_name}")

        print(f"Processing model: {model_name} -> {latex_file_path}")
        missing_files_notes = []
        processed_queries_for_model = [] 

        with open(latex_file_path, 'w') as latex_f:
            create_latex_header(latex_f, model_name)
            start_long_table(latex_f, model_name)
            
            num_actual_query_blocks_to_write = 0
            for qid_check in QUERY_IDS:
                check_path = os.path.join(input_model_dir, f"{model_name}_{qid_check}Ranking_Top10.dat")
                if os.path.exists(check_path):
                    try:
                        with open(check_path, 'r') as temp_rf:
                            if sum(1 for _line in temp_rf if _line.strip()) > 1:
                                 num_actual_query_blocks_to_write +=1
                    except Exception as e_scan:
                        print(f"    Scan warning: Could not read {check_path} during pre-scan: {e_scan}")
                        pass 
            
            query_blocks_written_so_far = 0

            for query_id in QUERY_IDS:
                ranking_file_name = f"{model_name}_{query_id}Ranking_Top10.dat"
                ranking_file_path = os.path.join(input_model_dir, ranking_file_name)

                if os.path.exists(ranking_file_path):
                    ranking_data = []
                    file_had_data_rows = False
                    try:
                        with open(ranking_file_path, 'r') as rf:
                            lines = rf.readlines()
                            if len(lines) <=1: 
                                print(f"    Info: File {ranking_file_path} has no data beyond header.")
                            for line_num, line in enumerate(lines):
                                if line_num == 0: continue 
                                parts = line.strip().split()
                                if len(parts) == 2:
                                    doc_id_val = parts[0]
                                    try:
                                        score_val = float(parts[1])
                                        ranking_data.append((doc_id_val, score_val))
                                        file_had_data_rows = True 
                                    except ValueError:
                                        print(f"    Warning: Could not parse score in line: '{line.strip()}' from {ranking_file_path}")
                                elif line.strip(): 
                                    print(f"    Warning: Skipping malformed line: '{line.strip()}' from {ranking_file_path}")
                        
                        if ranking_data: 
                            processed_queries_for_model.append(query_id)
                            if query_blocks_written_so_far > 0:
                                latex_f.write("\\midrule\n")
                            
                            write_ranking_rows_for_query(latex_f, query_id, ranking_data)
                            query_blocks_written_so_far += 1

                        elif not file_had_data_rows and os.path.exists(ranking_file_path): 
                            missing_files_notes.append(f"\\item Ranking file found ({ranking_file_name}), but no valid data parsed or file empty beyond header.")

                    except Exception as e:
                        missing_files_notes.append(f"\\item Error processing ranking file {ranking_file_name}: {str(e).replace('_', ' ')}.")
                        print(f"    Error reading or processing file {ranking_file_path}: {e}")
                else:
                    missing_files_notes.append(f"\\item Ranking file not found: {ranking_file_name}.")
            
            end_long_table(latex_f)

            if not processed_queries_for_model: 
                 latex_f.write(f"\\section*{{No Ranking Data Processed}}\\nNo ranking data was successfully parsed and written to the table for model {model_name.replace('_',' ')} from directory {input_model_dir.replace('_',' ')}.\\\\\n")

            if missing_files_notes:
                latex_f.write("\\section*{File Processing Notes}\n")
                latex_f.write("\\begin{itemize}\n")
                for note in missing_files_notes:
                    latex_f.write(f"{note}\n")
                latex_f.write("\\end{itemize}\n")

            create_latex_footer(latex_f)
        print(f"Finished generating LaTeX for {model_name}.")

if __name__ == "__main__":
    main()
    print("\\nScript execution finished.")
    print(f"Please check the '{LATEX_OUTPUT_DIR}' directory for the .tex files.") 