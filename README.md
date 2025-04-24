Company Classification Using Taxonomy

Objective
Assign the most appropriate label from a predefined insurance-industry taxonomy to each company by leveraging text (descriptions, tags, sector/category/niche fields).


In
## 🚀 Initial Approach

1. **Text Cleaning**  
   - Normalized with NLTK: lowercasing, non-alphanumeric removal, lemmatization.

2. **Taxonomy Expansion**  
   - Generated variants for each label: original, lowercase, lemmatized.

3. **Text Embeddings**  
   - Used `all-MiniLM-L6-v2` to embed company texts and label variants.

4. **Label Assignment**  
   - Computed cosine similarity and picked the label with the highest score.

### ⚠️ Limitations

- Short or vague descriptions led to noisy embeddings.  
- Pure similarity missed nuanced, contextual relationships.  
- Model biased toward generic labels.
The results were not sufficiently accurate.


## 🔄 Revised Approach

After seeing poor accuracy, the pipeline became:

1. **Label Context Enrichment**  
   - Generation of rich, human-readable descriptions for each taxonomy label via a prompt-based LLM.

2. **Embedding Enriched Labels**  
   - Re-embed those generated descriptions with `SentenceTransformer`.

3. **Embedding Companies**  
   - Concatenate `description`, `business_tags`, `sector`, `category`, `niche` into a single text and embed.

4. **Similarity Pre-selection (Top 20)**  
   - For each company, compute cosine similarity against *enriched* label embeddings and select the top 20 candidates.

5. **Zero-Shot Classification**  
   - Run the company text against those 20 labels using `facebook/bart-large-mnli`.  
   - Choose the label with the highest confidence score.

### 🔑 Key Benefits

- **Deeper Semantic Coverage** via enriched label descriptions  
- **Robust Matching** through a final zero-shot “reasoning” step  
- **Scalable** – pre-selection reduces the label space dramatically

## 📊 Results

Top 10 Labels
The top 10 most frequent predicted labels in the classification results are:

Accessory Manufacturing: 1549 occurrences

E-Commerce Services: 462 occurrences

Low-Rise Foundation Construction: 394 occurrences

Corporate Responsibility Services: 335 occurrences

Food Processing Services: 259 occurrences

Financial Services: 244 occurrences

Rendering Services: 231 occurrences

Community Engagement Services: 222 occurrences

Arts Services: 209 occurrences

Business Development Services: 191 occurrences

Low-Confidence Cases
Approximately 86.2% of the classifications were low-confidence cases, with similarity scores lower than 0.5. These cases are less certain and may need further refinement.

Examples of Low-Confidence Predictions
Here are some examples of low-confidence predictions (with similarity scores below 0.5):


Predicted Label	Score
Project Management Services	0.101575
Frozen Food Processing	0.257470
Low-Rise Foundation Construction	0.139197
Plastic Manufacturing	0.091761
Low-Rise Foundation Construction	0.260388

Possible Improvements
### 1. Fine-tuning the Embedding Model
The embedding model could be fine-tuned on a small set of manually labeled pairs (`company_description`, `correct_label`) (e.g. 200–300 examples). This would help the vectors capture domain-specific language nuances in the insurance industry.

### 2. Multi-label Classification
Support could be added for assigning multiple labels per company (e.g. both “Property Insurance” and “Risk Assessment Services”).  
Rather than forcing a single choice, the zero-shot step can return all labels whose confidence scores exceed a chosen threshold (e.g. > 0.5), storing them in a `predicted_labels` list.

### 3. Augmenting Label Descriptions
Label descriptions can be enriched with two or three example sentences illustrating each category, or by incorporating standardized industry terminology (e.g. NAICS/SIC codes).  
More detailed descriptions tend to produce higher-quality embeddings and reduce hallucinations from the LLM.

### 4. Active Learning / Human-in-the-Loop
An active-learning loop could be introduced by automatically flagging companies with low confidence (e.g. similarity < 0.4) for expert review.  
Expert corrections would then be fed back into the training set, continuously improving the model’s performance on difficult cases.

### 6. Using Specialized Business/Financial Embeddings
Specialized models such as `FinBERT`, `InsuranceBERT` (if available), or OpenAI’s `text-embedding-ada-002` embeddings may capture insurance-specific jargon more effectively.  
Benchmarking several embedding sources on a small validation set would identify the best performer.

