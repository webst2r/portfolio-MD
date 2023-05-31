import pandas as pd
import numpy as np


class RuleBasedClassifiers:
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.n_samples = 0
        self.n_features = 0
        self.csv_datasets = [
            "../../datasets/contact_lenses.csv",
            "../../datasets/restaurant_customer_rating_feats.csv",
            "../../datasets/mushrooms.csv",
            "../../datasets/2015_happiness_report_mod.csv",
            "../../datasets/biomechanical_orthopedic_feats.csv"
        ]
        self.csv_datasets_col_names = [
            ['age', 'visual_deficiency', 'astigmatism', 'production', 'lens'],
            ['user_id', 'place_id', 'rating', 'food_rating', 'service_rating'],
            ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
             'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
             'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
             'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
             'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat'],
            ['country', 'region', 'happiness_rank', 'happiness_score', 'standard_error',
             'economy_gdp', 'family', 'health_life_expectancy', 'freedom', 'trust_gov_corruption',
             'generosity', 'dystopia_residual'],
            ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
             'sacral_slope', 'pelvic_radius', 'spondylolisthesis_degree', 'diagnostic']
        ]

    def repair_continuous_attributes(self, dataset, features):
        # Converts continuous attributes from float to int
        self.n_samples = dataset.shape[0]
        self.n_features = dataset.shape[1] - 1

        for feature in features:
            if dataset[feature].dtype == np.float64:
                dataset[feature] = dataset[feature].astype(int)

    def load_csv_dataset(self, csv_path, feature_names):
        # Loads and returns a CSV dataset with the provided column names
        dataset = pd.read_csv(csv_path)
        dataset.columns = feature_names
        return dataset

    def fix_dataset_missing_values(self, dataset):
        # Replaces missing values with the most frequent value in each column
        for column in dataset.columns:
            dataset[column] = dataset[column].replace('?', np.NaN)
            dataset[column] = dataset[column].fillna(dataset[column].value_counts().index[0])

    def build_learning_sets(self, dataset, class_attr, train_size):
        # Builds the train/test sets based on the specified train size
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        n_train = int(self.n_samples * train_size)
        n_test = self.n_samples - n_train

        dataset_ = dataset.copy(deep=True)
        self.fix_dataset_missing_values(dataset_)

        print(dataset_)

        input("Continue")

        self.y_train = dataset_.loc[0:n_train, class_attr].copy(deep=True)
        self.y_test = dataset_.loc[n_train + 1:self.n_samples, class_attr].copy(deep=True)

        dataset_ = dataset_.drop(class_attr, axis=1)

        self.X_train = dataset_.loc[0:n_train].copy(deep=True)
        self.X_test = dataset_.loc[n_train + 1:self.n_samples].copy(deep=True)

    def data_preprocessing(self):
        # Performs data preprocessing steps
        print('A) ::Processing CSV files::')
        dataset = self.load_csv_dataset(self.csv_datasets[0], self.csv_datasets_col_names[0])

        print('B) ::Repairing continuous attributes in Dataset::')
        self.repair_continuous_attributes(dataset, dataset.columns)

        print('C) ::Building train/test sets::')
        self.build_learning_sets(dataset, dataset.columns[-1], 1.0)

    def PRISM(self):
        # Generates the PRISM rule set
        prism_rule_set = []
        for label in set(self.y_train):
            instances = [i for i, val in enumerate(self.y_train) if val == label]

            while instances:
                rule = []
                X_train_ = self.X_train.copy(deep=True)
                instances_covered = []
                perfect_rule = False
                rule_precision = 0.0
                rule_coverage = 0.0

                while not perfect_rule and len(rule) < self.n_features + 1:
                    optimal_selector = [("","")]
                    optimal_selector_prec = [0.0, 0.0, 0.0]
                    instances_covered = []

                    for attribute in X_train_.columns:
                        attr_column = X_train_.loc[:, attribute]

                        for attr_value in set(attr_column):
                            total_attr_values_instances = attr_column[(attr_column == attr_value)].index._values
                            total_matches = len(total_attr_values_instances)
                            positive_attr_values_instances = list(set(total_attr_values_instances) & set(instances))
                            positive_matches = len(positive_attr_values_instances)

                            precision = (1.0 * positive_matches) / total_matches
                            coverage = (1.0 * positive_matches) / self.n_samples

                            if precision > optimal_selector_prec[2]:
                                optimal_selector = (attribute, attr_value)
                                optimal_selector_prec[0] = positive_matches
                                optimal_selector_prec[1] = total_matches
                                optimal_selector_prec[2] = precision
                                rule_precision = precision
                                rule_coverage = coverage
                                instances_covered = positive_attr_values_instances
                            elif precision == optimal_selector_prec[2] and positive_matches > optimal_selector_prec[0]:
                                optimal_selector = (attribute, attr_value)
                                optimal_selector_prec[0] = positive_matches
                                optimal_selector_prec[1] = total_matches
                                optimal_selector_prec[2] = precision
                                instances_covered = positive_attr_values_instances
                                rule_precision = precision
                                rule_coverage = coverage

                    if 0.0 < optimal_selector_prec[2] < 1.0:
                        rule.append(optimal_selector)
                        selector = rule[-1]
                        filtered_rows = X_train_[(X_train_[selector[0]] != selector[1])].index._values
                        X_train_ = X_train_.drop(filtered_rows).copy(deep=True)
                        X_train_ = X_train_.drop(selector[0], axis=1)

                        if len(X_train_.columns) == 0:
                            perfect_rule = True
                            continue
                    elif optimal_selector_prec[2] == 1.0:
                        rule.append(optimal_selector)
                        perfect_rule = True
                        continue
                    elif optimal_selector_prec[2] == 0.0:
                        input("Unexpected case")

                instances = list(set(instances) - set(instances_covered))
                rule.append(label)
                rule.append([rule_precision, rule_coverage])

                print("++++++++ RULE FOUND +++++++++")
                metrics = rule[-1]
                print("Rule:")
                print(rule)
                print("Rule-Precision: " + str(metrics[0]))
                print("Rule-Coverage: " + str(metrics[1]))
                print("\n")

                prism_rule_set.append(rule)

        return prism_rule_set


def main():
    rbc = RuleBasedClassifiers()
    rbc.data_preprocessing()
    rule_set = rbc.PRISM()

    print("%%%%%%%%%%%%%%%%% FINAL PRISM RULE SET %%%%%%%%%%%%%%%%%")
    print("\n")
    for prism_rule in rule_set:
        print(prism_rule)


if __name__ == '__main__':
    main()
