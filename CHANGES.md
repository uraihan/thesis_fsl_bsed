# Change logs
---
### 4.12.2024 - 3:08
- Introducing Angular Margin Loss (AML) component for further test with Devset. This is one of two necessary components for building ACL. AML is computed against an embedding after the last pooling and before the linear layer. The encoder is feeded with a transformed data similar to the original SCL. 
- Unlike the SCL, the AML implementation only calculates one projection of the embedding.
- Next step in the test is to conduct ablation of AML to find the best margin value
