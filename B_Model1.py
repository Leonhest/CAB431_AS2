import NewsItem
import math
import Parser
import os
import re

def b_model1(coll, q_parsed):
    """
    Calculate documents BM25 score in collection using Okapi BM25 formula.

    Parameters:
    coll (dict): 
        A collection of documents.
        - keys (str): NewsItem ID.
        - values (NewsItem): NewsItem object.
    q_parsed (list or dict): 
        Parsed query.
        If list: it's a list of query terms (e.g., ['term1', 'term2']).
        If dict: it's a map of query terms to their frequencies (e.g., {'term1': 2, 'term2': 1}).

    Returns:
    bm25_results (dict): 
        A dictionary of BM25 scores for each document.
        - keys (NewsItem): NewsItem object from the input collection.
        - values (float): Calculated BM25 score for the NewsItem.
    """
    # BM25 constants
    k1 = 1.2
    k2 = 500
    b = 0.75

    bm25_results = {}
    N = len(coll)

    # Step 1: Process query input to get term frequencies (qf_i)
    q_freq_map = {} 
    if isinstance(q_parsed, list):
        for term in q_parsed:
            q_freq_map[term] = q_freq_map.get(term, 0) + 1
    elif isinstance(q_parsed, dict):
        q_freq_map = q_parsed

    # Step 2: Calculate average document length (avdl)
    total_doc_length = 0
    for news_item in coll.values():
        total_doc_length += news_item.getSize()
    
    avdl = total_doc_length / N

    # Step 3: Pre-calculate document frequency (n_i) for each term in the query
    doc_freq_n_i = {} 
    for term_i in q_freq_map.keys():
        count = 0
        for news_item in coll.values(): 
            if term_i in news_item.terms: 
                count += 1
        doc_freq_n_i[term_i] = count

    # Step 4: Calculate BM25 score for each document in the Dataset
    for news_item_obj in coll.values(): 
        dl = news_item_obj.getSize() 
        
        K = k1 * ((1 - b) + b * (dl / avdl))
        
        current_doc_score = 0.0

        for term_i, qf_i in q_freq_map.items():
            f_i = news_item_obj.terms.get(term_i, 0)
            n_i = doc_freq_n_i.get(term_i, 0)

            idf_numerator = N - n_i + 0.5
            idf_denominator = n_i + 0.5
            
            idf_i = 0.0
            if idf_denominator > 0 and (idf_numerator / idf_denominator) > -1:
                 idf_i = math.log(1 + (idf_numerator / idf_denominator))
            
            term_score_part1 = 0.0
            if (K + f_i) > 0:
                term_score_part1 = ((k1 + 1) * f_i) / (K + f_i)
            
            term_score_part2 = 0.0
            if (k2 + qf_i) > 0:
                term_score_part2 = ((k2 + 1) * qf_i) / (k2 + qf_i)
            
            term_contribution = idf_i * term_score_part1 * term_score_part2
            current_doc_score += term_contribution
            
        bm25_results[news_item_obj] = current_doc_score
         
    return bm25_results

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
                if len(parts) >= 1:
                    ranked_doc_ids.append(parts[0])

        # Step 2: Read relevance judgements 
        relevance = {} #
        total_relevant_in_truth = 0
        with open(relevance_judgments_file, 'r') as f_rel:
            for line in f_rel:
                parts = line.strip().split()
                doc_id = parts[1]
                rel_score = int(parts[2])
                relevance[doc_id] = rel_score
                if rel_score == 1:
                    total_relevant_in_truth +=1


        # Precision@10 
        relevant_at_10 = 0
        for i in range(min(10, len(ranked_doc_ids))):
            doc_id = ranked_doc_ids[i]
            if relevance.get(doc_id, 0) == 1:
                relevant_at_10 += 1
        precision_at_10 = relevant_at_10 / 10.0 if len(ranked_doc_ids) >= 10 else relevant_at_10 / len(ranked_doc_ids) if len(ranked_doc_ids) > 0 else 0.0


        # Average Precision (AP)
        ap_sum = 0.0
        relevant_docs_found = 0
        for i in range(len(ranked_doc_ids)):
            doc_id = ranked_doc_ids[i]
            if relevance.get(doc_id, 0) == 1:
                relevant_docs_found += 1
                precision_at_i = relevant_docs_found / (i + 1.0)
                ap_sum += precision_at_i
        
        average_precision = (ap_sum / total_relevant_in_truth) if total_relevant_in_truth > 0 else 0.0

        # Discounted Cumulative Gain @10 (DCG@10)
        dcg_score = 0.0
        num_ranked_docs = len(ranked_doc_ids)

        doc_id_at_rank_1 = ranked_doc_ids[0]
        rel_1 = float(relevance.get(doc_id_at_rank_1, 0))
        dcg_score += rel_1

        p_cutoff = 10
        effective_p = min(p_cutoff, num_ranked_docs)

        for rank_i in range(2, effective_p + 1): 
            doc_id_at_rank_i = ranked_doc_ids[rank_i - 1] 
            rel_i = float(relevance.get(doc_id_at_rank_i, 0))
            
            if rel_i > 0: 
                dcg_score += rel_i / math.log2(rank_i) 

        return average_precision, precision_at_10, dcg_score

    except FileNotFoundError:
        print(f"Error: Ranking or relevance file not found. Sys: {system_ranking_file}, Rel: {relevance_judgments_file}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return 0.0, 0.0, 0.0

if __name__ == '__main__':
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    with open('Queries-2.txt', 'r') as f:
        queries_text = f.read()

    main_results_dir = "results"
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)

    # Changed directory names to reflect B_Model1
    output_dir_b_model1 = os.path.join(main_results_dir, "Outputs_B_Model1")
    output_dir_top10_b_model1 = os.path.join(main_results_dir, "Outputs_Top10_B_Model1")

    if not os.path.exists(output_dir_b_model1):
        os.makedirs(output_dir_b_model1)
    
    if not os.path.exists(output_dir_top10_b_model1):
        os.makedirs(output_dir_top10_b_model1)


    all_ap = []
    all_p10 = []
    all_dcg10 = []
    processed_query_count = 0

    # Changed metrics file name to reflect B_Model1
    output_metrics_file = os.path.join(main_results_dir, "evaluation_results_B_Model1.txt")

    with open(output_metrics_file, 'w') as metrics_f:
        for dataset_num in range(101, 151):
            dataset_folder = f"DataSets/Dataset{dataset_num}"
            
            if not os.path.exists(dataset_folder):
                continue

            news_collection = Parser.News_Parser(stop_words, dataset_folder)

            query_pattern = rf"<num>\s*Number:\s*R{dataset_num}\s*<title>\s*(.*?)\s*<desc>\s*Description:\s*(.*?)\s*<narr>"
            query_match = re.search(query_pattern, queries_text, re.DOTALL)
            
            if query_match:
                query_title = query_match.group(1).strip()
                query_desc = query_match.group(2).strip()
                full_query = f"{query_title} {query_desc}"
                parsed_query = Parser.Q_Parser(full_query, stop_words)
                
                b_model1_scores_map = b_model1(news_collection, parsed_query)
                
                # Output filenames use the updated directory paths
                output_filename = os.path.join(output_dir_b_model1, f"B_Model1_{dataset_num}Ranking.dat")
                output_filename_top10 = os.path.join(output_dir_top10_b_model1, f"B_Model1_{dataset_num}Ranking_Top10.dat")
                
                sorted_docs = sorted(b_model1_scores_map.items(), key=lambda item: item[1], reverse=True)
                
                with open(output_filename, 'w') as out_f:
                    for news_item_obj, score in sorted_docs:
                        doc_id = news_item_obj.getNewsId()
                        out_f.write(f"{doc_id} {score:.6f}\n") 

                with open(output_filename_top10, 'w') as out_f_top10:
                    out_f_top10.write(f"Query{dataset_num} (DocID BM25_Score):\n")
                    for news_item_obj, score in sorted_docs[:10]:
                        doc_id = news_item_obj.getNewsId()
                        out_f_top10.write(f"{doc_id} {score:.6f}\n")

                # Evaluation
                relevance_judgments_path = f"RelevanceJudgements/Dataset{dataset_num}.txt"
                if os.path.exists(relevance_judgments_path):
                    ap, p10, dcg10 = evaluate_ranking(output_filename, relevance_judgments_path)
                    all_ap.append(ap)
                    all_p10.append(p10)
                    all_dcg10.append(dcg10)
                    processed_query_count +=1
                    metric_line = f"Query R{dataset_num}: AP={ap:.4f}, P@10={p10:.4f}, DCG@10={dcg10:.4f}"
                    print(metric_line)
                    metrics_f.write(metric_line + "\n")
                else:
                    warning_line = f"Warning: Relevance judgment file not found for R{dataset_num}: {relevance_judgments_path}"
                    print(warning_line)
                    metrics_f.write(warning_line + "\n")

            else:
                fail_line = f"Failed to find query for dataset {dataset_num} (R{dataset_num})"
                print(fail_line)
                metrics_f.write(fail_line + "\n")
                
        # Calculate and Print Average Metrics
        if processed_query_count > 0:
            map_score = sum(all_ap) / processed_query_count
            avg_p10 = sum(all_p10) / processed_query_count
            avg_dcg10 = sum(all_dcg10) / processed_query_count

            overall_header = "\n--- Overall Evaluation Results ---"
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
            no_eval_line = "\nNo queries were processed for evaluation."
            print(no_eval_line)
            metrics_f.write(no_eval_line + "\n")

    print(f"\nProcessing complete. Output files are in '{main_results_dir}' directory.")
    print(f"Evaluation metrics also saved to {output_metrics_file}")