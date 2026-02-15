#!/usr/bin/env python3
"""
Benchmark script to compare stemmer output with correct answers.
Computes accuracy and edit distance accuracy between answer.txt and output.txt.
"""

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def edit_distance_accuracy(s1, s2):
    """
    Calculate edit distance accuracy between two strings.
    Accuracy = 1 - (edit_distance / max_length)
    """
    if not s1 and not s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def load_file(filename):
    """
    Load lines from a file, stripping whitespace.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []


def compute_metrics(answer_file, output_file, input_file=None):
    """
    Compare two files and compute accuracy metrics.
    """
    correct_answers = load_file(answer_file)
    user_outputs = load_file(output_file)
    input_words = load_file(input_file) if input_file else None

    min_lines = min(len(correct_answers), len(user_outputs))
    exact_matches = 0
    total_edit_distance_accuracy = 0.0
    correct_reduction_total = 0.0
    user_reduction_total = 0.0
    
    true_positives = 0 
    false_positives = 0 
    false_negatives = 0 
    
    for i in range(min_lines):
        correct = correct_answers[i]
        output = user_outputs[i] if i < len(user_outputs) else ""
        original = input_words[i] if input_words and i < len(input_words) else ""
        
        is_exact_match = correct == output
        if is_exact_match:
            exact_matches += 1
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
        
        edit_acc = edit_distance_accuracy(correct, output)
        total_edit_distance_accuracy += edit_acc
        
        if original and input_words:
            correct_reduction = ((len(original) - len(correct)) / len(original)) * 100 if len(original) > 0 else 0
            user_reduction = ((len(original) - len(output)) / len(original)) * 100 if len(original) > 0 else 0
            correct_reduction_total += correct_reduction
            user_reduction_total += user_reduction
    
    exact_accuracy = (exact_matches / min_lines) * 100 if min_lines > 0 else 0
    avg_edit_accuracy = (total_edit_distance_accuracy / min_lines) * 100 if min_lines > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_correct_reduction = correct_reduction_total / min_lines if min_lines > 0 and input_words else 0
    avg_user_reduction = user_reduction_total / min_lines if min_lines > 0 and input_words else 0
    
    print(f"\nResults Summary:")
    print(f"================")
    print(f"Total lines compared: {min_lines}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_accuracy:.2f}%")
    print(f"Average edit distance accuracy: {avg_edit_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-measure: {f_measure:.4f}")
    
    if input_words:
        print(f"Average correct reduction ratio: {avg_correct_reduction:.2f}%")
        print(f"Average user reduction ratio: {avg_user_reduction:.2f}%")
        print(f"Reduction ratio difference: {abs(avg_correct_reduction - avg_user_reduction):.2f}%")

def main():
    """
    Main function to run the benchmark.
    """
    compute_metrics('answer_root.txt', 'output_root.txt', 'input.txt')


if __name__ == "__main__":
    main()