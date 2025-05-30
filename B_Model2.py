import math
import os
from nltk.stem import PorterStemmer
import NewsItem
from B_Model1 import evaluate_ranking
import re


def b_model2(query, stop_words, inputfolder):
    """
    Ranks documents using the query liklihood algorithm given a query

    Parameters:
    query (string): Sentence used to search for documents
    stop_words (list of string): List of common english words
    inputfolder (string): Path to folder containing the news item document dataset

    Returns:
    results (dict): Document frequencty dictionary
        - keys (string): NewsItem ID
        - values (float): Ranking score for the NewsItem.
    """

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
                    #append term to termSet
                    try:
                        termSet[word] += 1
                    except:
                        termSet[word] = 1

            item.setSize(size)

        documentColl[item.getNewsId()] = item

    #Calculate Rank Score
    result = {} #Dictionary to contain document scores
    C = sum(termSet.values()) #Size of termset
    la = 0.4 #Lambda variable

    for docId, newsItem_obj in documentColl.items():
        d_terms = newsItem_obj.get_termList() #List terms of document
        d_size = newsItem_obj.getSize() #Size of document
        score = 0.0

        if not queryDict:
            result[docId] = score
            continue

        for q_term in queryDict.keys():
            f_qi_d = d_terms.get(q_term, 0)
            
            c_qi = termSet.get(q_term, 0)
            
            combined_prob = (1 - la) * (f_qi_d / d_size) + la * (c_qi / C)
            
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

            b_model2_scores = b_model2(current_query_string, stop_words, input_folder_path)
            
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
