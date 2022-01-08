import sys
import math


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
        #zasto je tu math.ceil kod total time a kod total distance nije
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
        #self.route.pop(index)

        #new_route = Route(self.route[0][0])

        #for el in self.route:
        #    new_route.add(el[0])


        #self.route = new_route[::]

        #return self.route

        new_route = Route(self.route[0][0])
        for i, customer in enumerate([x[0] for x in self.route][1:]):
            if i != index - 1:
                new_route.add(customer)
        return new_route

    def insert(self, customer, index, add_possible=[True, True]):
        new_route1 = Route(self.route[0][0])
        new_route2 = Route(self.route[0][0])

        for i, customer_i in enumerate([x[0] for x in self.route][1:]):
            if i == index - 1:
                if add_possible[0] and new_route1.check_adding_constraints(customer):
                    new_route1.add(customer)
                else:
                    add_possible[0] = False

            if add_possible[0] and new_route1.check_adding_constraints(customer_i):
                new_route1.add(customer_i)
            else:
                add_possible[0] = False

            if add_possible[1] and new_route2.check_adding_constraints(customer_i):
                new_route2.add(customer_i)
            else:
                add_possible[1] = False

            if i == index - 1:
                if add_possible[1] and new_route2.check_adding_constraints(customer):
                    new_route2.add(customer)
                else:
                    add_possible[1] = False

        if add_possible[0] and add_possible[1]:
            if new_route1.route[-1][1] < new_route2.route[-1][1]:
                return new_route1
            else:
                return new_route2

        if add_possible[0]:
            return new_route1
        if add_possible[1]:
            return new_route2

        return None

    def __repr__(self):
        output_string = ''

        for el in self.route[:-1]:
            output_string += f'{el[0].index}({el[1]})->'

        output_string += f'{self.route[-1][0].index}({self.route[-1][1]})\n'

        return output_string


class Solution:
    def __init__(self):
        self.n_routes = 0
        self.total_time = 0
        self.total_distance = 0
        self.routes = []

    def get_sorted_routes(self):
        return sorted(self.routes, key=lambda x: len(x.route))

    def add(self, route):
        self.routes.append(route)
        self.n_routes += 1
        self.total_time += route.total_time
        self.total_distance += route.total_distance

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

        """
        # sort by due date ASC
        customers_sorted_dt = sorted(customers, key=lambda x: x.due_date)

        # sort by ready time ASC
        customers_sorted_rt = sorted(customers_sorted_dt, key=lambda x: x.ready_time)

        # sort by distance from depot ASC
        customers_sorted_depot_dist = sorted(customers_sorted_dt, key=lambda x: distance(depot, x))

        for customer in customers_sorted_depot_dist:
            if customer not in visited:
                first = customer
                break

        route.add(first)
        visited.add(first)

        customers_sorted_first_dist = sorted(customers_sorted_dt, key=lambda x: distance(first, x))
        for i in range(1, len(customers_sorted_first_dist)):
            customer = customers_sorted_first_dist[i]
            if route.check_adding_constraints(customer) and customer not in visited:
                route.add(customer)
                visited.add(customer)
            if route.remaining_capacity == 0:
                break
        route.add(depot)
        solution.add(route)
        """

    return solution


def get_neighbor(current_solution):
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


def local_search(current_solution, number_of_iterations=100):
    better_exists = True
    iter = 0

    while better_exists:
        current_solution, better_exists = get_neighbor(current_solution)
        #print(better_exists, iter, len(current_solution.get_sorted_routes()))
        print(f'{better_exists=}, {iter=}, n_routes={len(current_solution.get_sorted_routes())}, distance={current_solution.total_distance}')
        #print(current_solution)
        iter += 1
        if iter >= number_of_iterations:
            break
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
    #solution = local_search(solution)

    with open(sys.argv[2], "w") as out_file:
        out_file.write(f'{solution}')

    print(solution)
