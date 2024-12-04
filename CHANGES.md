# Change logs
---
### 4.12.2024 - 3:08
- Introducing Angular Margin Loss (AML) for further test with Devset. AML is computed against an embedding after the last pooling and before the linear layer. The encoder is feeded with a transformed data similar to the original SCL. 
- Unlike the SCL, the AML implementation only calculates one projection of the embedding.
- Next step in the test is to compare performance of SCL and AML, SCL and AML without transformed data, and SCL and AML where two projection of the embedding matrix are taken into AML calculation instead of one.
