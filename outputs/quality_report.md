# Adverse action reason quality report

Denials explained: 2375

Consistency (share of 5 nearest neighbours sharing the top reason): 0.694

Faithfulness (share of denials that flip to approval when the top reason feature is moved to an approved applicant level): 0.851

Coverage, first reason actionable: 0.771

Coverage, any of the four reasons actionable: 1.000

Reading the numbers. Consistency near 1 means similar applicants get similar explanations, which is what a reviewer expects. Faithfulness measures whether the stated reason actually drives the decision; a low number would mean the notices point at features the model does not really depend on. Coverage asks whether the applicant can do anything about the reasons given, which matters because a notice built from reasons the applicant cannot change (such as length of employment) is accurate but not useful.
