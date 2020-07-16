library(ggplot2)
library(dplyr)

df <- read.csv("condition_differences.csv")

df %>% filter(Condition.2 == "ne ko se") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="ne ko se") +
    ggsave("Results/animate_2000000/ne_ko_se_distr_diff.png")


df %>% filter(Condition.2 == "ne se ko") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="ne se ko") +
    ggsave("Results/animate_2000000/ne_se_ko_distr_diff.png")


df %>% filter(Condition.2 == "ko ne se") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="ko ne se") +
    ggsave("Results/animate_2000000/ko_ne_se_distr_diff.png")


df %>% filter(Condition.2 == "ko se ne") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="ko se ne") +
    ggsave("Results/animate_2000000/ko_se_ne_distr_diff.png")


df %>% filter(Condition.2 == "se ko ne") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="se ko ne") +
    ggsave("Results/animate_2000000/se_ko_ne_distr_diff.png")


df %>% filter(Condition.2 == "se ne ko") %>% 
    ggplot(mapping=aes(x=KL.Divergence, y=Intersection.Union, fill=Condition.1)) + 
    geom_point(colour="black", size=4, pch=21) + 
    scale_fill_manual(values=c("#2f4f4f","#7f0000","#008000","#000080","#ff8c00","#ffff00",
                               "#00ff00","#00ffff","#ff0000","#1e90ff","#f5deb3","#ff69b4")) +
    labs(x="KL divergence", y="Intersection/Union", fill="Real dists", title="se ne ko") +
    ggsave("Results/animate_2000000/se_ne_ko_distr_diff.png")
