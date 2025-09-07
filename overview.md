# NutriScope: Predicting Hidden Hunger Risk  

## Overall Approach  

NutriScope was designed with a single goal in mind: to make the invisible challenge of hidden hunger visible and actionable. Hidden hunger refers to micronutrient deficiencies that occur even when people appear to be adequately nourished. It affects billions worldwide and has profound consequences on health, education, and economic productivity, particularly in underserved and low-income communities.  

Our approach during TurBioHacks was to build a web application that brings together machine learning, intuitive visualization, and public health policy alignment. NutriScope allows users to input basic demographic and dietary information such as age, gender, income bracket, education level, and daily intake of Vitamin A, Vitamin D, Zinc, Iron, and Folate. This minimal input lowers barriers to participation while still providing enough detail to produce meaningful insights.  

The project pipeline began with **data preprocessing**. Demographic variables were encoded to be usable by classification models, and nutrient intake data was normalized to ensure fair comparisons across users. With this cleaned dataset, we experimented with multiple predictive models. A Logistic Regression model provided a baseline, while more complex approaches such as Random Forest, Gradient Boosting, and Neural Networks offered improved predictive accuracy. To evaluate performance, we relied on **F1 score** and **ROC-AUC**, balancing sensitivity and precision in identifying hidden hunger risks.  

The final product integrates these models into a lightweight but functional web application. Outputs are presented in both personal and community contexts. For the individual, NutriScope generates a **risk flag** alongside a nutrient gap analysis, showing which deficiencies are most influential. For the community, the app can aggregate anonymized user data into **geospatial heatmaps** that highlight vulnerable regions, allowing policymakers to better target interventions. This two-tier approach bridges the gap between personal health awareness and public health strategy.  

NutriScope is also intentionally aligned with broader solutions such as the *Triple Shield Policy*, which emphasizes fortification, supplementation, and education. By creating a screening tool that complements these proven interventions, we envision NutriScope as both a personal health assistant and a policy support system.  

---

## Key Insights  

One of the strongest insights from our work was the identification of **Vitamin A, Folate, and Zinc** as the top predictors of hidden hunger. Feature importance analysis consistently highlighted these nutrients, validating global research on their critical role in health outcomes. Their importance underscores why fortification and supplementation strategies often prioritize these three.  

Another key insight was the **dual-use potential** of NutriScope. On one hand, individuals can use the app to better understand their nutrition status and take small steps toward improvement. On the other, policymakers and organizations can leverage aggregated results to identify high-risk communities and allocate resources more effectively. This dual focus increases the impact of a single platform.  

We also learned the value of **clear data visualization**. Machine learning outputs are not inherently user-friendly. By designing heatmaps, nutrient breakdown charts, and simple binary risk indicators, we were able to transform raw predictions into insights that resonate with non-technical audiences. This translation of science into practice became one of the most rewarding aspects of the project.  

Finally, working within the hackathon timeframe provided an important lesson in **rapid prototyping**. We discovered that impactful tools can be built quickly when the focus is clear, the models are carefully chosen, and the output is streamlined. The speed of development did not compromise the depth of the work but instead highlighted the importance of prioritization and iterative design.  

---

## Takeaways  

The development of NutriScope reinforced several important lessons:  

- **Complex models can drive simple solutions.** Advanced machine learning techniques can be embedded in user-friendly interfaces that empower individuals without overwhelming them.  
- **Nutrition is both a personal and a public issue.** Hidden hunger must be addressed at the household level through awareness and at the population level through coordinated interventions.  
- **Digital tools can bridge knowledge gaps.** By translating datasets and statistical models into accessible insights, NutriScope demonstrates how technology can make invisible health challenges visible.  
- **Hackathons are fertile grounds for innovation.** The limited timeframe encouraged creativity, teamwork, and efficiency, all of which are transferable skills for future projects.  

Ultimately, NutriScope is more than just a web application. It is a proof of concept that shows how data science, thoughtful design, and public health vision can converge to address one of the world’s most pervasive nutrition challenges. By combining personalized insights with community-level data, NutriScope has the potential to grow into a long-term tool for healthier individuals and stronger, more resilient communities.  
