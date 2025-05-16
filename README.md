**Computational Cognitive Science Project** 

**Task 2** 

You are required to use Ant colony optimization to solve a variant of the traveling salesman problem.  In the variant, you have *n* cities. You have to start at a city (doesn’t matter which one) and visit all the  other cities and come back to the starting city once you’re finished using the lowest cost (or shortest  

path) possible. Making a full loop through all of the cities. You will be using ant colony optimization  (ACO) to tackle this problem. 

**Problem Configuration** 

You have to run the ACO on a given set of cities under different configurations. 

You’ll run ACO on a set of 10 cities with a different amount of ant agent each time. The given number  of ant agents you’ll use is 1, 5, 10, 20\. The amount of ant agents does not change in the run. Each run  consists of 50 iterations, a single iteration consists of having all the ant agents complete their full  loop (going through each city once and back to the start). After you finish, you’ll repeat the same  runs when the number of cities is 20 (on each given ant agent amount for each run). 

You will need to use the ants to figure out the shortest path going through all the cities and back to  the starting city. 

Generate the distances between the cities such that the path between a city is between 3 and 50 (inclusive and integers). Generate those distances only once (once for the 10 and once for the 20\)  and use the same generated distances for the different amount of ant agents.

**Code** 

The code should contain the generation function used and the generated distances for the set of 10  cities and the set of 20 (The simplest way is to use a 2D matrix to represent the from city x to city y  distance). 

Show the values used for the distances (you will need them for the runs after the generation, so keep  them in the code and show them) 

The code should contain the logic of the ants traveling finding their way according to the ACO, the  process and the extraction of the final solution. 

**Report** 

• First of all, explain how the ACO will work in the problem. Then show the chosen distances  between the cities in each given configuration. 

• Show the development of the pheromone map (This can be done by a graph or a table of  values where i , j is the path from city i to j)and current optimal path for every 10 iteration  (do not include iteration 0, start from 10). 

• Show the results relating to the set of 10 cities first in a separate section, then the set of 20  cities. 

• In each set of cities, include the results asked above per amount of ant agents (for the 1 ant  agent run, then the 5 ant agents run, etc…). 

• Comment on the progress and optimal solution of the runs of all the sets of ant agents at the  end of section of the city set and write your conclusion. 

• At the end of the report write if there were any differences when you used the same amount  of ant agents on a larger set of cities (10 cities vs 20 overall)? 

**Deliverables** 

• Code 

• Report 

• Presentation