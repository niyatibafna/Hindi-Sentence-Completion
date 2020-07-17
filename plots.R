library(ggplot2)
library(dplyr)
'%ni%' <- Negate ('%in%')

models.2np <- c("4gram_2np", "lossy_rand_erasure_2np", "lossy_pred_bias2_2np")

models <- c("4gram", "lossy_rand_erasure", "lossy_pred_bias", "lossy_pred_bias2", "human")
folder <- "Results/"

gram_df <- data.frame()
for (model in models) {
    df <- read.csv(paste("gram_", model, ".csv", sep=""))
    df$model <- model
    df$nnp <- 3
    gram_df <- gram_df %>% rbind (df)
}
for (model in models.2np) {
    print(model)
    df <- read.csv(paste("gram_", model, ".csv", sep=""))
    df$model <- substr(model, 1, nchar(model)-4)
    df$nnp <- 2
    gram_df <- gram_df %>% rbind (df)
}
ungram_df <- gram_df
ungram_df$Frequency <- 1 - ungram_df$Frequency


error_class_df <- data.frame()
for (model in models) {
    df <- read.csv(paste("error_class_", model, ".csv", sep=""))
    df$model <- model
    error_class_df <- error_class_df %>% rbind (df)
}

kl_div_df <- data.frame()
for (model in models) {
    if (model != "human") {
        df <- read.csv(paste("kl_div_", model, ".csv", sep=""))
        df$model <- model
        kl_div_df <- kl_div_df %>% rbind (df)
    }
}

cond_order <- c("ne-ko-se", "ne-se-ko", "ko-ne-se", "ko-se-ne", "se-ko-ne", "se-ne-ko")
model_order <- c("human", "4gram", "lossy_rand_erasure", "lossy_pred_bias2", "lossy_pred_bias")

gram_df$model <- factor(gram_df$model, levels=model_order)
ungram_df$model <- factor(ungram_df$model, levels=model_order)
error_class_df$model <- factor(error_class_df$model, levels=model_order)
kl_div_df$model <- factor(kl_div_df$model, levels=model_order)

gram_df %>% filter (model %ni% c("human", "lossy_pred_bias")) %>%
 group_by(model, nnp) %>% summarise(mean_gramm = mean(Frequency))

gram_df %>% filter((nnp == 3) & (model %in% c("4gram", "lossy_rand_erasure", "lossy_pred_bias2"))) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Condition, y=Percentage, fill=model)) + 
    # geom_point() + geom_line() + 
    geom_bar(position="dodge", stat="identity") +
    labs(x = "Condition", y = "Percentage Grammatical", fill="") +
    # theme(axis.text.x = element_text(size=6.7, angle = 0)) + #, vjust = 0.5, hjust=0.5)) +
    scale_fill_manual(labels = c("4-gram", "LC-Surp\nRand-Eras", "LC-Surp\nPred-Bias"), 
                    values = c("red", "blue", "green")) +
    theme(
        legend.position = c(.82, .99),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank()
    ) + 
    ggsave(paste(folder, "gram_models.png", sep=""), width=4.4, height=4, dpi=300, units="in")

ungram_df %>% filter((nnp == 3) & (model %in% c("human", "4gram"))) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Condition, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    labs(x = "Condition", y = "Percentage Ungrammatical", fill="") +
    # theme(axis.text.x = element_text(size=6.7, angle = 0)) + #, vjust = 0.5, hjust=0.5)) +
    scale_fill_manual(labels = c("Human", "4-gram"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.55, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank()
    ) + 
    ggsave(paste(folder, "gram_human_4gram.png", sep=""), width=4.4, height=4, dpi=300, units="in")

ungram_df %>% filter((nnp == 3) & (model %in% c("human", "lossy_rand_erasure"))) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Condition, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    labs(x = "Condition", y = "Percentage Ungrammatical", fill="") +
    # theme(axis.text.x = element_text(size=6.7, angle = 0)) + #, vjust = 0.5, hjust=0.5)) +
    scale_fill_manual(labels = c("Human", "LC-Surp\nRand-Eras"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.25, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank()
    ) + 
    ggsave(paste(folder, "gram_human_lossy_rand_erasure.png", sep=""), width=4.4, height=4, dpi=300, units="in")

ungram_df %>% filter((nnp == 3) & (model %in% c("human", "lossy_pred_bias2"))) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Condition, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    labs(x = "Condition", y = "Percentage Ungrammatical", fill="") +
    # theme(axis.text.x = element_text(size=6.7, angle = 0)) + #, vjust = 0.5, hjust=0.5)) +
    scale_fill_manual(labels = c("Human", "LC-Surp\nPred-Bias"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.25, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank()
    ) + 
    ggsave(paste(folder, "gram_human_lossy_pred_bias.png", sep=""), width=4.4, height=4, dpi=300, units="in")

# 
# 
error_class_df %>% filter(model %in% c("human", "4gram")) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Error.Type, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    facet_wrap(. ~ Condition) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(x = "Error Type", y = "Percentage", fill="", facet="Condition") +
    scale_fill_manual(labels = c("Human", "4-gram"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.85, .99),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank(),
        legend.text=element_text(size=7),
        legend.key.size = unit(0.6,"line")
    ) + 
    ggsave(paste(folder, "ec_human_4gram.png", sep=""), width=4.4, height=4, dpi=300, units="in")


error_class_df %>% filter(model %in% c("human", "lossy_rand_erasure")) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Error.Type, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    facet_wrap(. ~ Condition) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(x = "Error Type", y = "Percentage", fill="", facet="Condition") +
    scale_fill_manual(labels = c("Human", "LC-Surp\nRand-Eras"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.99, .99),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank(),
        legend.text=element_text(size=7),
        legend.key.size = unit(0.6,"line")
    ) + 
    ggsave(paste(folder, "ec_human_lossy_rand_erasure.png", sep=""), width=4.4, height=4, dpi=300, units="in")


error_class_df %>% filter(model %in% c("human", "lossy_pred_bias2")) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    mutate(Percentage=Frequency*100) %>%
    ggplot(mapping=aes(x=Error.Type, y=Percentage, fill=model)) + geom_bar(position="dodge", stat="identity") +
    facet_wrap(. ~ Condition) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(x = "Error Type", y = "Percentage", fill="", facet="Condition") +
    scale_fill_manual(labels = c("Human", "LC-Surp\nPred-Bias"), 
                    values = c("#3C6997", "#FF9B71")) +
    theme(
        legend.position = c(.99, .99),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2),
        legend.title=element_blank(),
        legend.text=element_text(size=7),
        legend.key.size = unit(0.6,"line")
    ) + 
    ggsave(paste(folder, "ec_human_lossy_pred_bias.png", sep=""), width=4.4, height=4, dpi=300, units="in")

# 
# 
kl_div_df %>% filter(model %in% c("4gram", "lossy_rand_erasure", "lossy_pred_bias2")) %>%
    mutate(Condition=factor(gsub(" ", "-", Condition), levels=cond_order)) %>%
    ggplot(mapping=aes(x=Condition, y=KLdiv_h_m, color=model, group=1)) + geom_point(size=3, alpha=0.7) + 
    geom_segment(mapping=aes(x=Condition, y=0, xend=Condition, yend=KLdiv_h_m)) +
    # geom_bar(stat="identity") + 
    labs(x = "Condition", y = "KL-Divergence (Human || Model)", color="Model", facet="Condition") +
    # theme(axis.text.x = element_text(size=6.7, angle = 0)) + #, vjust = 0.5, hjust=0.5)) +
    scale_color_manual(labels = c("4-gram", "LC-Surp Rand-Eras", "LC-Surp Pred-Bias"), 
                        values = c("green", "blue", "red")) +
    theme(
        legend.position = c(.65, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.margin = margin(2, 2, 2, 2)
    ) + 
    ggsave(paste(folder, "kl_div.png", sep=""), width=4.4, height=4, dpi=300, units="in")
