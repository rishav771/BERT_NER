# Product Attribute Extraction for Enhanced Online Shopping Experience

## Problem Statement

Understanding product attributes is crucial for improving customers' online shopping experience and constructing a comprehensive product knowledge graph. Existing methods focus on attribute extraction from text descriptions or utilize visual information from product images. However, detailed and accurate attribute values are essential for effective product search and recommendation, leading to better customer satisfaction, increased customer base, and higher revenue for e-commerce platforms. Therefore, accurate extraction of product attributes is crucial for enhancing the online shopping experience.

## Proposed Methodology

The proposed methodology aims to extract textual attributes from both product images and product text, utilizing cross-validation to improve accuracy. The methodology is divided into the following steps:

1. **Text Extraction via OCR**: Optical Character Recognition (OCR) is used to extract text content from product images. This step enhances the accuracy of attribute extraction by recovering missing information that may not be mentioned in the product title or description.

2. **Sequence Tagging on Text**: Sequence tagging, similar to Named Entity Recognition (NER), is applied to the product text (title, description) to identify and extract relevant named entities. This step involves tokenization, part-of-speech tagging, and named entity recognition, enabling the classification of attributes into predefined categories.

3. **Category-Specific Vocabulary**: Rather than using a fixed pre-defined vocabulary, the model dynamically switches between different category-specific vocabularies based on the category information of the input product. This approach ensures accurate extraction of domain-specific phrases and improves token selection accuracy.

4. **Cross-Validation of Results**: The extracted attributes from product titles and images are cross-validated to disambiguate multiple possible attribute values. The visual information from product images helps in identifying the correct attribute value when there are multiple possibilities mentioned in the product title.


## Example

Let's consider an example to illustrate the input and output formats:
```python
**Input**:
Product Text: "This shirt is a red Nike T-shirt available in size medium."

**Tokenized Input**:
["This", "shirt", "is", "a", "red", "Nike", "T", "-", "shirt", "available", "in", "size", "medium", "."]

**Attention Mask**:
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Output**:
Token-level Predictions: [O, O, O, O, COLOR, BRAND, BRAND, BRAND, BRAND, O, O, SIZE, SIZE, O]

Entity Spans: ["red Nike T-shirt", "medium"]

Entity Labels: ["COLOR", "SIZE"]

Entity Values: ["red Nike T-shirt", "medium"]
```
In this example, the model correctly predicts that "red Nike T-shirt" is the color of the product ("COLOR" entity label) and "medium" is the size of the product ("SIZE" entity label).

## Additional Approaches and Improvements

On top of the proposed methodology, two additional approaches were explored:

1. **OpenTag for Attribute Extraction**: One approach involved using OpenTag, a non-BERT-based model, to extract attribute values. While OpenTag provided valuable insights, the adoption of BERT-based token classification significantly improved the model's performance.

2. **BERT Token Classification and Fine-Tuning**: BERT-based token classification was employed for attribute extraction, and the model was fine-tuned to optimize performance on the attribute extraction task. This allowed the model to achieve better accuracy and generalization.

## Model Training and Deployment

To expedite the training process, the model was trained on multiple high-end GPUs, including Tesla A100, by leveraging distributed data parallelism. This approach significantly reduced the training time and allowed for the training of large models without sacrificing performance.

Utilizing AutoML for hyperparameter tuning further improved the model's performance by finding optimal hyperparameters that enhance attribute extraction accuracy.

The final trained model was deployed for real-world application, enabling efficient and accurate attribute extraction in a production environment. This deployment facilitated better product search and recommendation for customers, ultimately enhancing the overall online shopping experience.

## Conclusion

In conclusion, the proposed methodology, along with additional approaches and improvements, constitutes a powerful and effective solution for product attribute extraction in the e-commerce domain. By integrating multimodal data, employing BERT-based token classification, and leveraging distributed data parallelism, the model can accurately extract and classify product attributes, leading to improved customer satisfaction and business success.

---
*Note: This project's code and implementation details can be found in the respective directories of this repository.*
