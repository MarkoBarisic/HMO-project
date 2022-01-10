import sys
import math
import itertools
import random

# CLASSES
class Customer:
    def __init__(self, params):
        self.index = params[0]
        self.x = params[1]
        self.y = params[2]
        self.demand = params[3]
        self.ready_time = params[4]
        self.due_date = params[5]
        self.service_time = params[6]

    def __repr__(self):
        return f'Index:{self.index}\nX:{self.x}\nY:{self.y}\nDemand:{self.demand}\nReady time:{self.ready_time}\nDue date:{self.due_date}\nService timne:{self.service_time}\n'

    def __str__(self):
        return f'{self.index}-{self.x}-{self.y}-{self.demand}-{self.ready_time}-{self.due_date}-{self.service_time}'


class Route:
    def __init__(self, depot):
        self.total_time = 0
        self.total_distance = 0
        self.route = [(depot, 0)]
        self.remaining_capacity = vehicle_capacity


    def add(self, customer):
        self.total_time += math.ceil(distance(self.route[-1][0], customer))
        self.total_distance += distance(self.route[-1][0], customer)

        if self.total_time < customer.ready_time:
            self.total_time = customer.ready_time

        self.route.append((customer, self.total_time))

        self.total_time += customer.service_time
        self.remaining_capacity -= customer.demand


    def check_adding_constraints(self, customer):
        total_time = self.total_time + math.ceil(distance(self.route[-1][0], customer))

        #oce probit capacity constraint ili due date contraint
        if self.remaining_capacity < customer.demand or total_time > customer.due_date:
            return False

        if total_time < customer.ready_time:
            total_time = customer.ready_time

        #oce probit due date za povratak u depot
        return total_time + customer.service_time + math.ceil(distance(customer, self.route[0][0])) <= self.route[0][
            0].due_date


    def remove(self, index):
        if index == 0 or index == len(self.route) or len(self.route) == 3:
            return None

        new_route = Route(self.route[0][0])

        for i in range(1, len(self.route)):
            if i != index:
                new_route.add(self.route[i][0])

        return new_route


    def insert(self, customer, index):
        if index == 0:
            return None

        new_route = Route(self.route[0][0])

        i = 1
        while i < len(self.route):
            if i == index:
                cus = customer
                index = -1

            else:
                cus = self.route[i][0]
                i += 1

            if not new_route.check_adding_constraints(cus):
                return None

            new_route.add(cus)

        return new_route

    def permutations(self, start_i, size):
        perm_list = []

        for perm in itertools.permutations(range(start_i, start_i + size)):
            new_route = Route(self.route[0][0])

            cnt = 1
            valid = True
            while cnt < len(self.route):
                if cnt in perm:
                    for ind in perm:
                        if not new_route.check_adding_constraints(self.route[ind][0]):
                            break

                        else:
                            new_route.add(self.route[ind][0])
                    else:
                        cnt = max(perm) + 1
                        continue

                    valid = False
                    break

                else:
                    if not new_route.check_adding_constraints(self.route[cnt][0]):
                        valid = False
                        break

                    else:
                        new_route.add(self.route[cnt][0])
                        cnt += 1

            if valid:
                perm_list.append(new_route)

        return perm_list

    def __repr__(self):
        output_string = ''

        for el in self.route[:-1]:
            output_string += f'{el[0].index}({el[1]})->'

        output_string += f'{self.route[-1][0].index}({self.route[-1][1]})\n'

        return output_string

    def equals(self, route_2):
        if (len(self.route) != len(route_2.route)
            or self.total_time != route_2.total_time
            or self.total_distance != route_2.total_distance
            or self.remaining_capacity != route_2.remaining_capacity):
            return False

        for i in range(len(self.route)):
            if (self.route[i][0].index != route_2.route[i][0].index
                or self.route[i][1] != route_2.route[i][1]):
                return False

        return True


class Solution:
    def __init__(self):
        self.n_routes = 0
        self.total_time = 0
        self.total_distance = 0
        self.routes = []
        self.n_serverd_customers = 0

    def get_sorted_routes(self):
        return sorted(sorted(self.routes, key=lambda x: x.total_distance), key=lambda x: len(x.route))

    def add(self, route):
        self.routes.append(route)
        self.n_routes += 1
        self.total_time += route.total_time
        self.total_distance += route.total_distance
        self.n_serverd_customers += len(route.route)-2

    def shortest_route(self):
        return sorted(self.routes, key=lambda route: len(route.route))[0]

    def __repr__(self):
        output_string = f'{self.n_routes}\n'

        for i in range(len(self.routes)):
            output_string += f'{i + 1}: {self.routes[i]}'

        output_string += f'{self.total_distance}'

        return output_string


# FUNCTIONS
def parse_input(input_file):
    def parse_customer_line(line):
        return [int(x.strip()) for x in line.strip().split()]

    customers = []
    with open(input_file) as f:
        lines = f.readlines()
        vehicle_number, vehicle_capacity = parse_customer_line(lines[2])
        depot = Customer(parse_customer_line(lines[7]))
        for i in range(8, len(lines)):
            params = parse_customer_line(lines[i])
            customers.append(Customer(params))
    return vehicle_number, vehicle_capacity, depot, customers


def distance(c1, c2):
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

def index_of_nearest(customer, route):
    customers = [x[0] for x in route][1:-1]
    min = distance(customer, customers[0])
    ind = 0
    for i, customer2 in enumerate(customers):
        if min > distance(customer, customer2):
            min = distance(customer, customer2)
            ind = i
    return ind + 1


def greedy(depot, customers, n_vehicle):
    visited = set()
    solution = Solution()

    while len(visited) < len(customers) and solution.n_routes < n_vehicle:
        route = Route(depot)

        while True:
            # sort customers by due_date, ready_time, distance from last customer in that order
            eligible_customers = sorted(sorted(sorted(customers, key=lambda x: distance(route.route[-1][0], x)), key=lambda x: x.ready_time), key=lambda x: x.due_date)


            added = False
            for customer in eligible_customers:
                if  customer not in visited and route.check_adding_constraints(customer):
                    route.add(customer)
                    visited.add(customer)
                    added = True
                    break

            if route.remaining_capacity == 0 or not added:
                break

        route.add(depot)
        solution.add(route)

    if len(visited) < len(customers):
        print(len(visited))
        print(len(customers))
        for c in customers:
            if c not in visited:
                print(c)

    return solution


def get_neighbor(current_solution, operation, operation_1_mode='exh', operation_2_size=5):
    # create a neighborhood by applying neighbor operator to each route
    neighborhood = []

    current_sorted_routes = current_solution.get_sorted_routes()

    for route_index, route in enumerate(current_sorted_routes):

        if operation == 1:
            neighbor_candidates = []

            # neighbor = Solution()
            # 1) operator - take customer from a route and insert it into antoher route
            for insert_customer_index, insert_customer in enumerate(route.route[1:-1]):

                found_neighbor = False

                neighbor = Solution()

                # try to insert customer into route, start the search from largest routes
                for insert_route_index, insert_route in enumerate(current_sorted_routes[::-1]):
                    if route == insert_route:
                        # OVAJ DIO UTJECE NA i1, i2, i3 zele continue, i4, i5, i6 zeli break
                        if operation_1_mode=='quick':
                            break
                        if operation_1_mode=='exh':
                            continue

                    # check capacity constraint
                    if insert_route.remaining_capacity - insert_customer[0].demand < 0:
                        continue

                    # gather all version of a route with inserted customer
                    insert_route_versions = []

                    for insert_index in range(1, len(insert_route.route)):
                        modified_route = insert_route.insert(insert_customer[0], insert_index)

                        if modified_route:
                            #print('modified_route:', modified_route)
                            insert_route_versions.append(modified_route)

                    # if the customer can't be added try the next route
                    if not insert_route_versions:
                        continue

                    # else a neighbor is found, create a solution
                    # pick route version
                    if len(insert_route_versions) > 1:
                        insert_route_versions.sort(key=lambda x: x.total_distance)

                    for r_i, r in enumerate(current_sorted_routes):
                        if r_i == route_index:
                            route_rem = route.remove(insert_customer_index + 1)
                            if route_rem:
                                neighbor.add(route_rem)

                        elif r_i == current_solution.n_routes - 1 - insert_route_index:
                            neighbor.add(insert_route_versions[0])

                        else:
                            neighbor.add(r)

                    found_neighbor = True
                    break

                if found_neighbor:
                    neighbor_candidates.append(neighbor)


            if neighbor_candidates:
                neighbor_candidates.sort(key=lambda sol: sol.total_distance)
                neighbor_candidates.sort(key=lambda sol: sol.n_routes)

                neighborhood.append(neighbor_candidates[0])

                # dio koji utjece isto i4, i5, i6 zeli break
                if operation_1_mode=='quick':
                    break


        elif operation == 2:
            # 2) operator - rearange 3 sequential customers in a route
            if len(route.route) - 2 < 2:
                continue

            permutation_size = min(len(route.route) - 2, operation_2_size)

            for i in range(1, len(route.route) - permutation_size):
                neighbor = Solution()
                permutations = route.permutations(i, permutation_size)

                if permutations:
                    permutations.sort(key=lambda x: x.total_distance)

                    for r_i, r in enumerate(current_sorted_routes):
                        if r_i == route_index:
                            neighbor.add(permutations[0])

                        else:
                            neighbor.add(r)

                    neighborhood.append(neighbor)

        #elif operation == 3:
            # 3) operator - exchange customers between two routes
        #    for swap_route_index, swap_route in enumerate(current_sorted_routes):
        #        if route == swap_route:
        #            continue


    if neighborhood:
        neighborhood.sort(key=lambda sol: sol.total_distance)
        neighborhood.sort(key=lambda sol: len(sol.shortest_route().route))
        neighborhood.sort(key=lambda sol: sol.n_routes)

        improved_n_vehicles = neighborhood[0].n_routes < current_solution.n_routes
        improved_total_distance_only = neighborhood[0].n_routes == current_solution.n_routes and neighborhood[0].total_distance < current_solution.total_distance
        improved_shortest_route = len(neighborhood[0].shortest_route().route) < len(current_solution.shortest_route().route)

        if improved_n_vehicles or improved_total_distance_only or improved_shortest_route:
            return neighborhood[0], True

        else:
            return neighborhood[0], False

    else:
        return None, False
    (
    """
    current_solution_sorted = current_solution.get_sorted_routes()

    for i in range(len(current_solution_sorted)):
        for j in range(i + 1, len(current_solution_sorted)):
            for k in range(1, len(current_solution_sorted[i].route) - 1):

                index = index_of_nearest(current_solution_sorted[i].route[k][0], current_solution_sorted[j].route)
                new_route = current_solution_sorted[j].insert(current_solution_sorted[i].route[k][0], index, [True, True])

                if new_route:
                    current_solution_sorted[j] = new_route
                    route_after_remove = current_solution_sorted[i].remove(k)
                    if len(route_after_remove.route) > 2:
                        current_solution_sorted[i] = route_after_remove
                    else:
                        del current_solution_sorted[i]
                    solution = Solution()
                    for route in current_solution_sorted:
                        solution.add(route)
                    return solution, True

    for i in range(len(current_solution_sorted)):
        for j in range(i + 1, len(current_solution_sorted)):
            for k in range(1, len(current_solution_sorted[i].route) - 1):

                distance_before1 = current_solution_sorted[i].total_distance
                distance_before2 = current_solution_sorted[j].total_distance

                index = index_of_nearest(current_solution_sorted[i].route[k][0], current_solution_sorted[j].route)
                customer = current_solution_sorted[j].route[index][0]
                new_route2 = current_solution_sorted[j].remove(index)
                new_route2 = new_route2.insert(current_solution_sorted[i].route[k][0], index, add_possible=[True, False])
                new_route1 = current_solution_sorted[i].remove(k)
                new_route1 = new_route1.insert(customer, k, add_possible=[True, False])

                if new_route1 and new_route2 and \
                        distance_before1 + distance_before2 > new_route1.total_distance + new_route2.total_distance:
                    current_solution_sorted[i] = new_route1
                    current_solution_sorted[j] = new_route2
                    solution = Solution()
                    for route in current_solution_sorted:
                        solution.add(route)
                    return solution, True

    return current_solution, False
    """
    )

def local_search(current_solution, improving_only=True, max_iter=2000):
    iter = 1

    best_solution = current_solution

    operation_1_mode = 'exh' if current_solution.n_serverd_customers < 500 else 'quick'
    operation_2_size = 3

    while iter <= max_iter:
        new_best = False
        new_solutions = []

        for operation in range(1, 3):
            new_solution, improving = get_neighbor(current_solution, operation, operation_1_mode, operation_2_size)

            if improving:
                current_solution = new_solution
                break
        else:
            if new_solutions:
                current_solution = random.choice(new_solutions)
            else:
                if operation_2_size == 6:
                    exit()
                    
                operation_2_size += 1
                continue

        # check if current is new best
        if current_solution.n_routes < best_solution.n_routes:
            best_solution = current_solution
            new_best = True

        elif current_solution.n_routes == best_solution.n_routes:
            if current_solution.total_distance < best_solution.total_distance:
                best_solution = current_solution
                new_best = True

        if new_best:
            with open(sys.argv[2], "w") as out_file:
                out_file.write(f'{best_solution}')

        print(f'----------------------------\niter: {iter}\nserverd customers: {current_solution.n_serverd_customers}\n{current_solution}')
        iter += 1

    return current_solution


if __name__ == "__main__":
    n_vehicle, vehicle_capacity, depot, customers = parse_input(sys.argv[1])

    # print('Number of vehicles:', n_vehicle)
    # print('Vehicle capacity:', vehicle_capacity)
    # print('Depot:',depot)
    # print('Customers')
    # for c in customers:
    #     print(c)
    # print()

    solution = greedy(depot, customers, n_vehicle)
    print(f'Greedy\n{solution}')

    with open(sys.argv[2], "w") as out_file:
        out_file.write(f'{solution}')

    solution = local_search(solution, improving_only=False, max_iter=500)
