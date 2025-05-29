import math
import os
from nltk.stem import PorterStemmer
import NewsItem
from itertools import islice
from B_Model1 import evaluate_ranking
import re

def New_Model3(query, stop_words, inputfolder):

    files = [f"./{inputfolder}/{file}" for file in os.listdir(inputfolder)] #Save all file paths within the input folder into array
    stemmer = PorterStemmer() 
    documentColl = {} #Collection of NewsItem objects
    termSet= {} #Set of all terms in document collection
    
    queryDict = NewsItem.Assignment_1.Q_Parser(query, stop_words, stemmer)


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
    results = {}
    C = sum(termSet.values())
    la = 0.2

    for docId, newsItem_obj in documentColl.items():
        d_terms = newsItem_obj.get_termList()
        d_size = newsItem_obj.getSize()
        score = 0.0

        if not queryDict:
            results[docId] = score
            continue

        for q_term, q_freq in queryDict.items():
            f_qi_d = d_terms.get(q_term, 0)
            c_qi = termSet.get(q_term, 0)
            
            combined_prob = (1 - la) * (f_qi_d / d_size if d_size > 0 else 0) + la * (c_qi / C if C > 0 else 0)
            
            if combined_prob > 0:
                score += math.log10(combined_prob)

        results[docId] = score

    ## Step 5 top -k
    k = 15 #k-value
    Set_C = {}
    e = 1e-9 
    sorted_scores = sorted(results.items(), key=lambda item: item[1], reverse=True)
    Set_C = {}
    for docId, score in sorted_scores[:k]:
        Set_C[docId] = documentColl[docId]

    # step 6
    temp_vocabulary_set = set() 
    z = 10 #highest-probability words in doc (10-25 typical vaules)

    for docId, newsItem in Set_C.items():
        terms = newsItem.get_termList() 

        first_z_word_keys = list(islice(terms.keys(), z))
        temp_vocabulary_set.update(first_z_word_keys) 

    vocabulary = list(temp_vocabulary_set) 

    #step 7 Calculate the relevance model probabilities P (w|R)
    relevance_model = {}
    for word in vocabulary:
        score = 0

        for docId, newsItem in Set_C.items():
            terms = newsItem.get_termList()
            doc_size = newsItem.getSize() 

            # P(w|D) for Step 7
            pw_d = (terms.get(word, 0.0) + e) / doc_size if doc_size > 0 else e 

            # ∏ P(q_i|D) for Step 7
            query_product = 1.0
            for q_i in queryDict.keys():
                pq_d = (terms.get(q_i, 0.0) + e) / doc_size if doc_size > 0 else e
                query_product *= pq_d

            score += pw_d * query_product

        relevance_model[word] = score

    #step 8: Re-rank using Jelinek-Mercer
    results = {}

    for docId, newsItem in documentColl.items():
        d_terms = newsItem.get_termList()
        d_size = newsItem.getSize() 
        current_doc_rerank_score = 0.0

        for term_v in vocabulary:
            P_term_R = relevance_model.get(term_v, 0.0) # P(term|RelevanceModel)

            if P_term_R == 0.0: # If P(term|R) is 0, this term contributes nothing
                continue

            # Get term frequencies for Jelinek-Mercer calculation
            f_term_d = d_terms.get(term_v, 0.0)       # Frequency of term_v in current document d
            c_term_coll = termSet.get(term_v, 0.0)  # Frequency of term_v in the entire collection

            # Jelinek-Mercer smoothed probability P_jm(term_v|d)
            jm_prob_term_d = (1 - la) * (f_term_d / d_size if d_size > 0 else 0) + \
                             la * (c_term_coll / C if C > 0 else 0)
            
            if jm_prob_term_d > 0:
                current_doc_rerank_score += P_term_R * math.log10(jm_prob_term_d)
        
        results[docId] = current_doc_rerank_score

    return results


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

    # Define output directories for B_Model3 within the main results directory
    output_dir_b_model3 = os.path.join(main_results_dir, "Outputs_B_Model3")
    output_dir_top10_b_model3 = os.path.join(main_results_dir, "Outputs_Top10_B_Model3")

    if not os.path.exists(output_dir_b_model3):
        os.makedirs(output_dir_b_model3)
    if not os.path.exists(output_dir_top10_b_model3):
        os.makedirs(output_dir_top10_b_model3)

    all_ap = []
    all_p10 = []
    all_dcg10 = []
    processed_query_count = 0

    # Metrics file path within the main results directory
    output_metrics_file = os.path.join(main_results_dir, "evaluation_results_B_Model3.txt")

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

            b_model3_scores = New_Model3(current_query_string, stop_words, input_folder_path)
            
            # Output filenames within their respective subdirectories in results
            output_filename = os.path.join(output_dir_b_model3, f"B_Model3_{dataset_num}Ranking.dat")
            output_filename_top10 = os.path.join(output_dir_top10_b_model3, f"B_Model3_{dataset_num}Ranking_Top10.dat")
            
            sorted_docs = sorted(b_model3_scores.items(), key=lambda item: item[1], reverse=True)
            with open(output_filename, 'w') as out_f:
                for doc_id, score in sorted_docs:
                    out_f.write(f"{doc_id} {score:.6e}\n") 

            with open(output_filename_top10, 'w') as out_f_top10:
                out_f_top10.write(f"QueryR{dataset_num} (DocID B_Model3_Score):\n")
                for doc_id, score in sorted_docs[:10]:
                    out_f_top10.write(f"{doc_id} {score:.6e}\n")

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

            overall_header = "\n--- Overall Evaluation Results (B_Model3) ---"
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
            no_eval_line = "\nNo queries were processed for B_Model3 evaluation."
            print(no_eval_line)
            metrics_f.write(no_eval_line + "\n")

    print(f"\nProcessing complete for B_Model3. Output files are in '{output_dir_b_model3}' and '{output_dir_top10_b_model3}' folders.")
    print(f"Evaluation metrics for B_Model3 also saved to {output_metrics_file}")