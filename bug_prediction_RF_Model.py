import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import ast
import networkx as nx
import requests


def fetch_commits(repo_owner, repo_name, access_token):
    # GitHub API URL for fetching commits
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
    
    # Headers for authentication
    headers = {
        "Authorization": f"token {access_token}"
    }
    
    # Send GET request to GitHub API
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        commits = response.json()
        commit_list = []
        
        # Extract commit IDs from the response
        for commit in commits:
            commit_id = commit['sha']
            commit_list.append(commit_id)
        
        return commit_list
    else:
        print(f"Failed to fetch commits: {response.status_code}")
        return None


def calculate_cyclomatic_complexity_of_source_code(source_code):
    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Create a directed graph to represent the control flow
    cfg = nx.DiGraph()

    # Define a visitor class to traverse the AST and build the control flow graph
    class CFGVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_node = 'start'
            cfg.add_node(self.current_node)

        def visit_FunctionDef(self, node):
            self.current_node = node.name
            cfg.add_node(self.current_node)
            cfg.add_edge('start', self.current_node)
            self.generic_visit(node)

        def visit_If(self, node):
            test_node = f"test_{node.lineno}"
            cfg.add_node(test_node)
            cfg.add_edge(self.current_node, test_node)

            # Visit the body of the if statement
            self.current_node = test_node
            self.visit(node.body[0])

            # Visit the else part if it exists
            if node.orelse:
                else_node = f"else_{node.lineno}"
                cfg.add_node(else_node)
                cfg.add_edge(test_node, else_node)
                self.current_node = else_node
                self.visit(node.orelse[0])

        def visit_Return(self, node):
            return_node = f"return_{node.lineno}"
            cfg.add_node(return_node)
            cfg.add_edge(self.current_node, return_node)

    # Create an instance of the visitor and visit the AST
    visitor = CFGVisitor()
    visitor.visit(tree)

    # Calculate cyclomatic complexity using the formula CC = E - N + 2P
    E = cfg.number_of_edges()
    N = cfg.number_of_nodes()
    P = 1  # Assuming a single connected component
    cyclomatic_complexity = E - N + 2 * P

    return cyclomatic_complexity

def get_commit_list_of_repo():
    pass

def get_commit_details():
    pass

# Load historical bug data and code metrics
data = pd.read_csv('/Users/fkhanamzm/Downloads/sample_bug_data.csv')


# Data cleaning
data.dropna(inplace=True)

# Feature selection
features = ['lines_of_code', 'cyclomatic_complexity', 'developer_activity']
X = data[features]
y = data['bug_present']

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc}')