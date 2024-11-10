import openai
import math

# Set up OpenAI API key
openai.api_key = "API_KEY"  # Replace with your OpenAI API key

prefix = "It is a truth universally acknowledged"
expected_suffix = "that a single man in possession of a good fortune, must be in want of a wife."
token_nums = len(expected_suffix.split(" "))





# Step 1: Generate Text with Log Probabilities using OpenAI's API

def generate_text_with_logprobs(prefix, max_tokens=50, temperature=0.3,top_k = 5):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  # Replace with the correct model
        prompt=prefix,
        max_tokens=max_tokens,  # Ensure this is an integer
        logprobs=top_k,  # Enables token-level log probabilities
        temperature=temperature
    )
    generated_text = response.choices[0].text.strip()
    log_probs = response.choices[0].logprobs.token_logprobs
    return generated_text, log_probs


# Step 2: Precision, Recall, and Hamming Distance Functions
def precision(expected, generated):
    expected_tokens, generated_tokens = expected.split(), generated.split()
    matched_tokens = sum(1 for g, e in zip(generated_tokens, expected_tokens) if g == e)
    return matched_tokens / len(generated_tokens) if generated_tokens else 0


def recall(expected, generated):
    expected_tokens, generated_tokens = expected.split(), generated.split()
    matched_tokens = sum(1 for g, e in zip(generated_tokens, expected_tokens) if g == e)
    return matched_tokens / len(expected_tokens) if expected_tokens else 0


def hamming_distance(generated, ground_truth):
    # Tokenize the generated and ground truth strings
    generated_tokens = generated.split()
    ground_truth_tokens = ground_truth.split()

    # Ensure both lists have the same length for token-level comparison
    if len(generated_tokens) != len(ground_truth_tokens):
        if len(generated_tokens) - len(ground_truth_tokens) > 0:
            generated_tokens = generated_tokens[0:len(ground_truth_tokens)]
        else:
            generated_tokens = generated_tokens + ["dummy" for i in range(-len(generated_tokens) + len(ground_truth_tokens))]
        # raise ValueError("Generated and ground truth sentences must have the same length for Hamming distance.")

    # Calculate the Hamming distance
    mismatches = sum(1 for x, g in zip(generated_tokens, ground_truth_tokens) if x != g)
    hamming_distance = mismatches / len(generated_tokens)  # Normalized by length

    return hamming_distance


# Step 3: Calculate Perplexity from Log Probabilities
def calculate_perplexity(log_probs):
    # Check for placeholders and filter them out
    valid_log_probs = [log_prob for log_prob in log_probs if log_prob != -9999.0]

    # Warn if there are missing values
    if len(valid_log_probs) < len(log_probs):
        print("Warning: Placeholder values detected and ignored in perplexity calculation.")

    # Calculate the average log probability and perplexity
    avg_log_prob = sum(valid_log_probs) / len(valid_log_probs)
    perplexity = math.exp(-avg_log_prob)
    return perplexity


def testing(prefix,expected_suffix,token_nums,top_k = 3,temperature=0.3):


    # Step 4: Run Generation and Evaluation
    try:
        # Generate text with log probabilities
        generated_text, log_probs = generate_text_with_logprobs(prefix, max_tokens=token_nums,temperature=temperature,top_k=top_k)

        generated_text = ' '.join(generated_text.strip().split(" ")[:token_nums])

        # generated_text = generated_text[:token_nums].strip()
        # generated_suffix = generated_text[len(prefix):].strip()

        # Calculate metrics
        mp = precision(expected_suffix, generated_text)
        mr = recall(expected_suffix, generated_text)
        mh = hamming_distance(expected_suffix, generated_text)
        perplexity = calculate_perplexity(log_probs)

        # Display results
        print("Original Prefix:", prefix)
        print("Generated Suffix:", generated_text)
        # print("Generated Suffix:", generated_suffix)
        print("Precision (MP):", mp)
        print("Recall (MR):", mr)
        print("Hamming Distance (MH):", mh)
        print("Perplexity:", perplexity)

    except Exception as e:
        print(f"Error: {e}")

    print("----------------------------------------------------------------")


if __name__ == "__main__":
    # testing(prefix, expected_suffix, token_nums, temperature=1.2)
    parameters = [(0.0,1),(0.3,3),(0.6,4),(0.9,4),(1.2,6)]

    for i,data in enumerate(parameters):
        t,j = data
        for k in range(j):
            print("----------------------------------------------------------------")
            print(f"with temperature {t:.4f} and generate {k:.0f} times")
            testing(prefix,expected_suffix,token_nums,temperature=t)
