import math
import os
from nltk.stem import PorterStemmer
import NewsItem
import re

def evaluate_ranking(system_ranking_file, relevance_judgments_file):
    """
    Evaluates a ranking system output against relevance judgments.

    Parameters:
    system_ranking_file (str): Path to the system's ranking output file.
                               Format: doc_id score (one per line, ranked)
    relevance_judgments_file (str): Path to the relevance judgments file.
                                   Format: query_id doc_id relevance_score

    Returns:
    tuple: (average_precision, precision_at_10, dcg_at_10)
           Returns (0,0,0) if files cannot be read or in case of errors.
    """
    try:
        # Step 1: Read system ranking
        ranked_doc_ids = []
        with open(system_ranking_file, 'r') as f_rank:
            for line in f_rank:
                parts = line.strip().split()
                if len(parts) >= 1: # Make sure there is at least a doc_id
                    ranked_doc_ids.append(parts[0])

        # Step 2: Read relevance judgements 
        relevance = {} 
        total_relevant_in_truth = 0
        with open(relevance_judgments_file, 'r') as f_rel:
            for line in f_rel:
                parts = line.strip().split()
                # Expecting: query_id doc_id relevance_score
                if len(parts) >= 3:
                    doc_id = parts[1]
                    try:
                        rel_score = int(parts[2])
                        relevance[doc_id] = rel_score
                        if rel_score > 0: # Assuming any score > 0 is relevant for counting total
                            total_relevant_in_truth +=1
                    except ValueError:
                        print(f"Warning: Could not parse relevance score for {doc_id} in {relevance_judgments_file}")
                        continue # Skip this line
                else:
                    print(f"Warning: Malformed line in {relevance_judgments_file}: {line.strip()}")


        # Precision@10 
        relevant_at_10 = 0
        num_docs_to_check_p10 = min(10, len(ranked_doc_ids))
        if num_docs_to_check_p10 > 0:
            for i in range(num_docs_to_check_p10):
                doc_id = ranked_doc_ids[i]
                if relevance.get(doc_id, 0) > 0: # Assuming rel > 0 means relevant
                    relevant_at_10 += 1
            precision_at_10 = relevant_at_10 / float(num_docs_to_check_p10)
        else:
            precision_at_10 = 0.0


        # Average Precision (AP)
        ap_sum = 0.0
        relevant_docs_found = 0
        if total_relevant_in_truth == 0:
             average_precision = 0.0
        else:
            for i in range(len(ranked_doc_ids)):
                doc_id = ranked_doc_ids[i]
                if relevance.get(doc_id, 0) > 0: # Assuming rel > 0 means relevant
                    relevant_docs_found += 1
                    precision_at_i = relevant_docs_found / (i + 1.0)
                    ap_sum += precision_at_i
            average_precision = (ap_sum / total_relevant_in_truth) if total_relevant_in_truth > 0 else 0.0

        # Discounted Cumulative Gain @10 (DCG@10)
        dcg_score = 0.0
        num_ranked_docs = len(ranked_doc_ids)

        if num_ranked_docs > 0 :
            # DCG_1
            doc_id_at_rank_1 = ranked_doc_ids[0]
            rel_1 = float(relevance.get(doc_id_at_rank_1, 0)) # Use actual relevance score
            dcg_score += rel_1

            # DCG for ranks 2 to 10
            p_cutoff = 10
            effective_p = min(p_cutoff, num_ranked_docs)

            for rank_i in range(2, effective_p + 1): 
                doc_id_at_rank_i = ranked_doc_ids[rank_i - 1] 
                rel_i = float(relevance.get(doc_id_at_rank_i, 0)) # Use actual relevance score
                
                if rel_i > 0: 
                    dcg_score += rel_i / math.log2(rank_i) 

        return average_precision, precision_at_10, dcg_score

    except FileNotFoundError:
        print(f"Error: Ranking or relevance file not found. Sys: {system_ranking_file}, Rel: {relevance_judgments_file}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"An error occurred during evaluation: {e} (Sys: {system_ranking_file}, Rel: {relevance_judgments_file})")
        return 0.0, 0.0, 0.0

def B_Model2(query, stop_words, inputfolder):

    files = [f"./{inputfolder}/{file}" for file in os.listdir(inputfolder)] #Save all file paths within the input folder into array
    stemmer = PorterStemmer() 
    documentColl = {} #Collection of NewsItem objects
    termSet= {} #Set of all terms in document collection
    
    queryDict = NewsItem.Assignment_1.Q_Parser(query, stop_words, stemmer) #Query document language


    for filename in files:
        item = NewsItem.NewsItem() #Initialize new NewsItem object
        docId = NewsItem.Assignment_1.getDocId(filename) 
        item.setNewsId(docId)
        text = NewsItem.Assignment_1.getTextSec(filename)
    
        size = 0 #Initaialize size of NewsItem document

        for line in text.splitlines():
            line = NewsItem.Assignment_1.cleanLine(line) #clean the line

            for word in line.split(): #Tokenization
                size += 1
                word = word.lower() #Lower case the word
                word = stemmer.stem(word) #Stem the word

                if (not word in stop_words and len(word) > 2):
                    item.add_term(word)
                    #term into big C
                    try:
                        termSet[word] += 1
                    except:
                        termSet[word] = 1

            item.setSize(size)

        documentColl[item.getNewsId()] = item

    ####################################
    result = {}
    total_term_occurrences_in_collection = sum(termSet.values())
    la = 0.2

    for docId, newsItem_obj in documentColl.items():
        d_terms = newsItem_obj.get_termList()
        d_size = newsItem_obj.getSize()
        score = 0.0

        if not queryDict:
            result[docId] = score
            continue

        for q_term, q_freq in queryDict.items():
            f_qi_d = d_terms.get(q_term, 0)
            
            prob_term_given_doc_model = 0.0
            if d_size > 0:
                prob_term_given_doc_model = f_qi_d / d_size
            
            coll_freq_of_q_term = termSet.get(q_term, 0)

            prob_term_given_coll_model = 0.0
            if total_term_occurrences_in_collection > 0:
                prob_term_given_coll_model = coll_freq_of_q_term / total_term_occurrences_in_collection
            
            combined_prob = (1 - la) * prob_term_given_doc_model + la * prob_term_given_coll_model
            
            if combined_prob > 0:
                score += math.log10(combined_prob)

        result[docId] = score

    return result


if __name__ == '__main__':
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    with open('Queries-2.txt', 'r') as f:
        queries_text_content = f.read() # Read all queries

    # Create the main results directory
    main_results_dir = "results"
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)

    # Define output directories for B_Model2 within the main results directory
    output_dir_b_model2 = os.path.join(main_results_dir, "Outputs_B_Model2")
    output_dir_top10_b_model2 = os.path.join(main_results_dir, "Outputs_Top10_B_Model2")

    if not os.path.exists(output_dir_b_model2):
        os.makedirs(output_dir_b_model2)
    if not os.path.exists(output_dir_top10_b_model2):
        os.makedirs(output_dir_top10_b_model2)

    all_ap = []
    all_p10 = []
    all_dcg10 = []
    processed_query_count = 0

    # Metrics file path within the main results directory
    output_metrics_file = os.path.join(main_results_dir, "evaluation_results_B_Model2.txt")

    with open(output_metrics_file, 'w') as metrics_f:
        # Process each dataset from 101 to 150
        for dataset_num in range(101, 151):
            current_dataset_id_str = f"R{dataset_num}"
            dataset_folder_name = f"Dataset{dataset_num}"
            input_folder_path = f"DataSets/{dataset_folder_name}"
            
            if not os.path.isdir(input_folder_path):
                warning_line = f"Warning: Dataset folder not found {input_folder_path}, skipping {current_dataset_id_str}."
                print(warning_line)
                metrics_f.write(warning_line + "\n")
                continue

            query_pattern = rf"<num>\s*Number:\s*{current_dataset_id_str}\s*<title>\s*(.*?)\s*<desc>\s*Description:\s*(.*?)\s*<narr>"
            query_match = re.search(query_pattern, queries_text_content, re.DOTALL)
            
            current_query_string = ""
            if query_match:
                query_title = query_match.group(1).strip()
                query_desc = query_match.group(2).strip()
                current_query_string = f"{query_title} {query_desc}"
            else:
                fail_line = f"Failed to find query for {current_dataset_id_str}"
                print(fail_line)
                metrics_f.write(fail_line + "\n")
                continue 

            if not current_query_string.strip():
                fail_line = f"Empty query for {current_dataset_id_str}, skipping."
                print(fail_line)
                metrics_f.write(fail_line + "\n")
                continue

            b_model2_scores = B_Model2(current_query_string, stop_words, input_folder_path)
            
            # Output filenames within their respective subdirectories in results
            output_filename = os.path.join(output_dir_b_model2, f"B_Model2_{dataset_num}Ranking.dat")
            output_filename_top10 = os.path.join(output_dir_top10_b_model2, f"B_Model2_{dataset_num}Ranking_Top10.dat")
            
            sorted_docs = sorted(b_model2_scores.items(), key=lambda item: item[1], reverse=True)
            
            with open(output_filename, 'w') as out_f:
                for doc_id, score in sorted_docs:
                    out_f.write(f"{doc_id} {score:.6f}\n") 

            with open(output_filename_top10, 'w') as out_f_top10:
                out_f_top10.write(f"QueryR{dataset_num} (DocID B_Model2_Score):\n")
                for doc_id, score in sorted_docs[:10]:
                    out_f_top10.write(f"{doc_id} {score:.6f}\n")

            relevance_judgments_path = f"RelevanceJudgements/{dataset_folder_name}.txt"
            if os.path.exists(relevance_judgments_path):
                ap, p10, dcg10 = evaluate_ranking(output_filename, relevance_judgments_path)
                all_ap.append(ap)
                all_p10.append(p10)
                all_dcg10.append(dcg10)
                processed_query_count +=1
                metric_line = f"Query {current_dataset_id_str}: AP={ap:.4f}, P@10={p10:.4f}, DCG@10={dcg10:.4f}"
                print(metric_line)
                metrics_f.write(metric_line + "\n")
            else:
                warning_line = f"Warning: Relevance judgment file not found for {current_dataset_id_str}: {relevance_judgments_path}"
                print(warning_line)
                metrics_f.write(warning_line + "\n")
            
        # Calculate and Print Average Metrics
        if processed_query_count > 0:
            map_score = sum(all_ap) / processed_query_count
            avg_p10 = sum(all_p10) / processed_query_count
            avg_dcg10 = sum(all_dcg10) / processed_query_count

            overall_header = "\n--- Overall Evaluation Results (B_Model2) ---"
            map_line = f"Mean Average Precision (MAP): {map_score:.4f}"
            avg_p10_line = f"Average Precision@10 (P@10): {avg_p10:.4f}"
            avg_dcg10_line = f"Average Discounted Cumulative Gain@10 (DCG@10): {avg_dcg10:.4f}"
            
            print(overall_header)
            print(map_line)
            print(avg_p10_line)
            print(avg_dcg10_line)

            metrics_f.write(overall_header + "\n")
            metrics_f.write(map_line + "\n")
            metrics_f.write(avg_p10_line + "\n")
            metrics_f.write(avg_dcg10_line + "\n")
        else:
            no_eval_line = "\nNo queries were processed for B_Model2 evaluation."
            print(no_eval_line)
            metrics_f.write(no_eval_line + "\n")

    print(f"\nProcessing complete for B_Model2. Output files are in '{output_dir_b_model2}' and '{output_dir_top10_b_model2}' folders.")
    print(f"Evaluation metrics for B_Model2 also saved to {output_metrics_file}")
