setwd("/Users/thulanihewavithana/Documents/PhD/Courses/CMPT898/CMPT-PLSC_819_Project/Dataset4//")
library(ggplot2)
library(reshape2)

# Assuming each file contains ancestry proportions for a given K
admixture_data <- read.table("pruned_data_admixture.6.Q")  # Replace with actual file name
colnames(admixture_data) <- paste0("Cluster_", 1:ncol(admixture_data))

# Reshape data for plotting
admixture_data$Individual <- 1:nrow(admixture_data)
melted_data <- melt(admixture_data, id.vars="Individual")

# Plot
strip_plot <- ggplot(melted_data, aes(x=Individual, y=value, fill=variable)) +
  geom_bar(stat="identity", width=1) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle=90, hjust=1, vjust=0.5, size=15), # Increase x-axis text size
    axis.text.y = element_text(size=15), # Increase y-axis text size
    axis.title.x = element_text(size=18, face="bold"), # Increase x-axis label size
    axis.title.y = element_text(size=18, face="bold"), # Increase y-axis label size
    legend.text = element_text(size=18), # Increase legend text size
    legend.title = element_text(size=16, face="bold"), # Increase legend title size
    panel.grid = element_blank(),  # Remove grid for cleaner appearance
    plot.margin = margin(10, 10, 10, 10),  # Adjust margins
    legend.position = "bottom",  # Position legend at the bottom
    legend.direction = "horizontal",  # Arrange legend items in a single row
    legend.box = "horizontal"  # Ensures the legend box is horizontal
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +  # Forces legend items into a single row
  labs(x="Individual", y="Ancestry Proportion", fill="Cluster")



# Save the plot as a strip-style figure
ggsave("admixture_strip_plot_legend_right.png", strip_plot, width=12, height=3, dpi=300)

