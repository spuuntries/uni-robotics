// Manually controlled

#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main()
{
    int m, n, finish[2], start[2];
    cin >> m >> n;
    map<pair<int, int>, pair<int, int>> parents;
    vector<pair<int, int>> queue;
    vector<vector<char>> matrix;

    for (int i = 0; i < m; i++)
    {
        vector<char> row;
        for (int j = 0; j < n; j++)
        {
            char temp;
            char specials[2] = {'$', '*'};
            cin >> temp;

            if (temp == specials[0])
            {
                finish[0] = i;
                finish[1] = j;
            }
            else if (temp == specials[1])
            {
                start[0] = i;
                start[1] = j;
            }

            row.push_back(temp);
        }
        matrix.push_back(row);
    }

    vector<vector<char>> dupl = matrix; // Duplicate to print result later

    // Start from the starting position
    queue.push_back(make_pair(start[0], start[1]));

    while (queue.size() > 0)
    {
        auto current = queue.front();
        queue.erase(queue.begin());

        vector<pair<int, int>> neighbors;

        if (current.first == finish[0] && current.second == finish[1]) // Reached finish, do path reconstruction on duplicate
        {
            vector<pair<int, int>> path;
            while (current.first != start[0] || current.second != start[1]) // Reconstruct path
            {
                int parentFirst = parents[make_pair(current.first, current.second)].first;
                int parentSecond = parents[make_pair(current.first, current.second)].second;
                path.push_back(make_pair(parentFirst, parentSecond));
                current.first = parentFirst;
                current.second = parentSecond;
            }

            for (int i = 0; i < path.size(); i++)
            {
                dupl[path[i].first][path[i].second] = '#';
            }

            dupl[finish[0]][finish[1]] = '*';

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    cout << dupl[i][j];
                    if (j == n - 1)
                        cout << endl;
                }
            }

            return 0;
        }

        // Checks left
        if (current.second != 0 && (matrix[current.first][current.second - 1] == '.' || matrix[current.first][current.second - 1] == '$'))
            neighbors.push_back(make_pair(current.first, current.second - 1));
        // Checks right
        if (current.second != matrix[0].size() - 1 &&
            (matrix[current.first][current.second + 1] == '.' || matrix[current.first][current.second + 1] == '$'))
            neighbors.push_back(make_pair(current.first, current.second + 1));
        // Checks top
        if (current.first != 0 && (matrix[current.first - 1][current.second] == '.' || matrix[current.first - 1][current.second] == '$'))
            neighbors.push_back(make_pair(current.first - 1, current.second));
        // Checks bottom
        if (current.first != matrix.size() - 1 &&
            (matrix[current.first + 1][current.second] == '.' || matrix[current.first + 1][current.second] == '$'))
            neighbors.push_back(make_pair(current.first + 1, current.second));

        if (neighbors.size() > 0)
        {
            for (int i = 0; i < neighbors.size(); i++)
            {
                queue.push_back(make_pair(neighbors[i].first, neighbors[i].second));
                parents[make_pair(neighbors[i].first, neighbors[i].second)] = make_pair(current.first, current.second);
                matrix[neighbors[i].first][neighbors[i].second] = '#';
                // Using marking logic because tracking a separate visited data structure would end up too complex.
            }
        }
    }

    cout << "Can't find valid path :C" << endl;
    return 0;

    /* Debugging input
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i][j];
            if (j == n - 1)
                cout << endl;
        }
    }

    for (int i = 0; i < sizeof(finish) / sizeof(finish[0]); i++)
    {
        cout << finish[i] << " ";
    }

    cout << endl;

    for (int i = 0; i < sizeof(current) / sizeof(current.first); i++)
    {
        cout << current[i] << " ";
    }

    cout << endl;
    */
}