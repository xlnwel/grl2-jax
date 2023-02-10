## HAPPO with Looking-Ahead v4

Looking-ahead policies are used to collect data for $\pi$ optimization.

We use the same permutation for both looking-ahead and ego optimization.

The first agent to update performs a fake looking-ahead update. That is, its updated policy is taken into account when computing the teammate ratio for HAPPO update but the parameters are not stored.
