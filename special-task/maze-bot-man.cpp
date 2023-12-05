// Manual version

#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main()
{
    int m, n, finish[2], start[2];
    cin >> m >> n;
    map<pair<int, int>, pair<int, int>> parents;
    vector<vector<char>> matrix;
    vector<char> queue;

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

    // Get input sequence
    int ins;
    cin >> ins;
    for (int i = 0; i < ins; i++)
    {
        char inchr;
        cin >> inchr;
        queue.push_back(inchr);
    }

    vector<pair<int, int>> processed;
    int current[2];
    int index = 0;
    current[0] = start[0];
    current[1] = start[1];

    while (queue.size() > 0)
    {
        char inst;
        inst = queue.front();
        queue.erase(queue.begin());

        switch (inst)
        {
        case 's':
            if (current[0] + 1 > m - 1)
                cout << index << " Robot Menabrak Tembok" << endl;
            else if (matrix[current[0] + 1][current[1]] == 'o')
                cout << index << " Robot Menabrak Obstacle" << endl;
            else
            {
                current[0]++;
                processed.push_back(make_pair(current[0], current[1]));
            }
            break;
        case 'w':
            if (current[0] - 1 < 0)
                cout << index << " Robot Menabrak Tembok" << endl;
            else if (matrix[current[0] - 1][current[1]] == 'o')
                cout << index << " Robot Menabrak Obstacle" << endl;
            else
            {
                current[0]--;
                processed.push_back(make_pair(current[0], current[1]));
            }
            break;
        case 'a':
            if (current[1] - 1 < 0)
                cout << index << " Robot Menabrak Tembok" << endl;
            else if (matrix[current[0]][current[1] - 1] == 'o')
                cout << index << " Robot Menabrak Obstacle" << endl;
            else
            {
                current[1]--;
                processed.push_back(make_pair(current[0], current[1]));
            }
            break;
        case 'd':
            if (current[1] + 1 > n - 1)
                cout << index << " Robot Menabrak Tembok" << endl;
            else if (matrix[current[0]][current[1] + 1] == 'o')
                cout << index << " Robot Menabrak Obstacle" << endl;
            else
            {
                current[1]++;
                processed.push_back(make_pair(current[0], current[1]));
            }
            break;
        default:
            break;
        }

        index++;
    }

    dupl[start[0]][start[1]] = '#';

    for (int i = 0; i < processed.size(); i++)
    {
        dupl[processed[i].first][processed[i].second] = '#';
        if (i == processed.size() - 1)
            dupl[processed[i].first][processed[i].second] = '*';
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << dupl[i][j];
            if (j == n - 1)
                cout << endl;
        }
    }

    if (processed.back() == make_pair(finish[0], finish[1]))
    {
        cout << "Robot Mencapai Finish" << endl;
        return 0;
    }

    cout << "Robot Tidak Mencapai Finish" << endl;
    return 0;
}