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
        self.route=[(depot, 0)]

    def add(self, customer):
        self.total_time += distance(self.route[-1][0], customer)

        if self.total_time < customer.ready_time:
            self.total_time = customer.ready_time

        self.route.append((customer, self.total_time))

        self.total_time += customer.service_time

    def __repr__(self):
        output_string = ''

        for el in self.route[:-1]:
            output_string += f'{el[0]}({el[1]})->'

        output_string += f'{route[-1][0]}({route[-1][1]})\n'

        return output_string


class Solution:
    def __init__(self):
        self.n_routes = 0
        self.total_time = 0
        self.routes = []

    def add(self, route):
        self.routes.append(route)
        self.n_routes += 1
        self.total_time += route.total_time

    def __repr__(self):
        output_string = f'{self.n_routes}\n'

        for i in range(len(self.routes)):
            output_string += f'{i + 1}: {self.routes[i]}\n'

        output_string += f'{self.total_time}'

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
    print(c1)
    print(c2)
    return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)


if __name__ == "__main__":
    n_vehicle, vehicle_capacity, depot, customers = parse_input(sys.argv[1])

    print('Number of vehicles:', n_vehicle)
    print('Vehicle capacity:', vehicle_capacity)
    print('Depot:',depot)
    print('Customers')
    for c in customers:
        print(c)
