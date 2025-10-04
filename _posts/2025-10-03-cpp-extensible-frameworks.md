---
layout: post
title: "Writing Extensible and Modular Frameworks in C++"
excerpt: "Great frameworks and libraries are the foundation of all C++ applications. In this post, I'll go over a few approaches to designing and implementing truly extensible and modular frameworks in C++!"
comments: true
---

When we're working in industry, we often have to build frameworks to consolidate duplicated code for re-use and help extend our code to future use-cases so we spend less time on integration and more time on business logic. When I work on completely novel problems, it's often faster to first work on the business logic without worrying about building a framework first since my priority is to get somtehing working first. Trying to generalize too early in an unknown problem space often causes more problems than it solves. Eventually, we converge to a point where the cost of properly designing a framework is lower than the cost of continuing to move forward wit hbespoke solutions for each problem.

As an example, in robotics, at the lowest level we usually have motion controllers operating on feedback loops on the order of hundreds of Hertz but we always have some kind of higher level on top that performs some kind of collision-aware local motion planning on top of those controllers. Going a layer above that, we often have planners or behaviors or actions (or whatever name they're given) that compose different local behaviors together via an interface. Starting out, getting the robot up and running with some planners is more important than trying to build a generic framework for those not-yet-existent planners. But after some critical point, building the planning framework helps accelerate adding new planners or fixing bugs in the existing ones.

There's never a right answer when building frameworks: it all depends on what the goal is and the state of the architecture. For example, do we know all of the plugins at compile time? Or do we not even know which plugins exist for a single invocation of our program? One example use-case for the latter would be graphics: if we're running on different Linux systems, we might want to load one kind of graphics engine vs. another depending on the specifics of the system. If we're building software for a robot, then we may want to load different libraries depending on the specific kinds of sensors.

The intention of this post is to explore some of those different kinds of approaches to extensible architectures by building a little Linux system monitor application as a motivating toy example. This system monitor will print out some useful information like CPU usage, CPU temperature, RAM usage, and uptime that will refresh at a fixed rate. We'll start with building the system moniter by getting everything working and printing to the screen first. Then we'll start adding in more considerations as we go and we'll see a few different options on how we can modify the existing architecture to make it more extensible. For example, suppose we want to isolate the executable that runs the monitors from the monitors themselves but we can't modify the monitor runner; how could we modify our architecture to support that use-case? (That's a sneak peek towards the direction we'll go!) We'll also look at a few ways to accomplish the same kind of extensibility using purely compile-time constructs.

Keep in mind that the focus is on the extensibility, not on the exact implementation details of the system monitor so I'll be a bit lax in terms of the implementations of the individual monitors, e.g., ignoring proper error handling, not performing proper testing, glossing over writing documentation.

Aside: my main motivation for writing this is my mild disappointment in the kinds of "extensible" architectures that I see in all kinds of code. The primary authors of these frameworks and libraries write out in their Readmes or in threads that "of course the framework is extensible and modular!" but then when I look through the effort it takes to create a new subcomponent, it requires things like "remember to add your component to this giant global registry!". This is not extensible to me, and there are better options that we'll explore!

All of the code will be available on my GitHub [here](https://github.com/mohitd/monitr)!

# Linux System Monitor

Now let's start building our system monitor! We'll be monitoring CPU usage, RAM usage, CPU temperature, and uptime. When the monitor runs, we'll print and format all of these out to the terminal and refresh those values at a fixed rate.

# MVP

The first thing we'll do is set up our minimum viable product (MVP). If we were developing a prototype or a small-scope throwaway project, then starting with a barebones MVP without a super strict design is often the fastest way to get something working. We'll follow that same route to start and then generalize it. That being said, even for this MVP, we'll still group logical blocks into functions. Let's start with setting up our `main` function and refresh rate (1s):

```cpp
void clear_screen() { std::cout << "\033[2J\033[1;1H"; }

int main(int, char**) {
    using namespace std::chrono_literals;

    while (true) {
        clear_screen();

        std::cout << "---- Linux System Monitor ----n\n";
        // monitor stuff goes here!

        std::this_thread::sleep_for(1s);
    }
    return 0;
}
```

As the name implies, `clear_screen()` clears the screen using some Unix terminal special characters so we'll refresh the screen and write new values of the monitors.

Let's start with the CPU monitor! On most Linux systems, information about the CPU is in the `/proc/stat` file. We're going to define a helper struct to match the format of this file so we can simply read it in using normal file stream operations. (On a Linux system, these aren't actually "files" like files on disk but "virtual" files that live in memory so it's completely fine to open and read from these at a "fast" rate since the OS isn't actually reading from the disk. It fits into the Linux philosophy that "everything is a file"!)

```cpp
struct CpuTimes {
    long long user{};     // Time spent in user mode.
    long long nice{};     // Time spent in user mode with low priority (nice).
    long long system{};   // Time spent in system mode.
    long long idle{};     // Time spent in the idle task.
    long long iowait{};   // Time waiting for I/O to complete.
    long long irq{};      // Time servicing interrupts.
    long long softirq{};  // Time servicing softirqs.
    long long steal{};    // Stolen time
};

CpuTimes read_cpu_times() {
    CpuTimes times{};
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return times;
    }

    std::string line;
    std::getline(stat_file, line);
    std::string cpu_label;
    std::stringstream ss(line);
    ss >> cpu_label >> times.user >> times.nice >> times.system >> times.idle
            >> times.iowait >> times.irq >> times.softirq >> times.steal;
    return times;
}
```

Now we'll need to compute the actual usage over an interval of time. One naïve way to calculate this is to look at the percentage of time that all CPUs are not idle (100% utilization minus idle time divided by the total time). We'll multiply by 100 since the calculation is in 1/100's of a Hz to get a percentage. Since we're already refreshing the screen at 1s, let's use that same rate (there's an implied division by the delta time but it's just 1 in this case). Let's wrap this in a function to calculate the CPU usage given a previous and next set of CPU times.

```cpp
double calculate_cpu_usage(const CpuTimes& prev, const CpuTimes& current) {
    const long long prev_idle = prev.idle + prev.iowait;
    const long long idle = current.idle + current.iowait;

    const long long prev_non_idle = prev.user + prev.nice + prev.system
                                    + prev.irq + prev.softirq + prev.steal;
    const long long non_idle = current.user + current.nice + current.system
                               + current.irq + current.softirq + current.steal;

    const long long prev_total = prev_idle + prev_non_idle;
    const long long total = idle + non_idle;

    const long long total_d = total - prev_total;
    const long long idle_d = idle - prev_idle;

    if (total_d == 0) {
        return 0.0;
    }

    // multiply by 100 since these are in 1/100s of Hz
    return 100. * (1.0 - (static_cast<double>(idle_d) / total_d));
}
```

Let's incorporate this into the `main` function. We'll need to keep track of the previous CPU times and when we're executing the loop for the first time so we don't print anything out.

```cpp
int main(int, char**) {
    using namespace std::chrono_literals;

    CpuTimes prev_times{};
    bool is_first_loop = true;
    //...
}
```

Now we'll need some special logic inside of the refresh loop to keep track of the previous and current CPU times and set `is_first_loop`.

```cpp
while (true) {
    clear_screen();

    const CpuTimes current_times = read_cpu_times();
    const double cpu_usage = calculate_cpu_usage(prev_times, current_times);

    std::cout << "---- Linux System Monitor ----\n";
    if (is_first_loop) {
        std::cout << std::format("{:<12} Calculating...\n", "CPU Usage:");
        is_first_loop = false;
    } else {
        std::cout << std::format("{:<12} {:>3.1f} %\n",
                                    "CPU Usage:", cpu_usage);
    }

    prev_times = current_times;
    std::this_thread::sleep_for(1s);
}
```

We fetch the current CPU times and use the previous one to calculate the CPU usage but if it's the first time in the loop, `prev_times` will be empty so we skip over it. (This could be more efficient if we don't bother calculating the CPU usage if we know it's the first time in the loop, but we'll make that minor optimization in the later sections.) Note we're using the new `std::format` in C++20 but everything we're doing can be accomplished by using stream operators or even `printf`. We'll do some minor formatting like `{:<12}` just to make the output line up and look a bit more tabular but it's just for aesthetics.

The other monitors are even simpler since we *can* get instantaneous results. For example, let's compute the RAM usage. Similar to the CPU usage, there's a file `/proc/meminfo` with the RAM usage that we can read in a similar fashion as the CPU times: we'll define a struct with the data we need and just read the relevant parts of that file into that struct.

```cpp
struct RamInfo {
    double total_gb{};
    double used_gb{};
    double percentage{};
};
```

This file is a bit more complicated than the one with CPU statistics but still easy to parse.

```cpp
RamInfo get_ram_usage() {
    std::ifstream meminfo_file("/proc/meminfo");
    std::string line;
    long mem_total{};
    long mem_available{};

    RamInfo info;
    if (!meminfo_file.is_open()) {
        return info;
    }

    while (std::getline(meminfo_file, line)) {
        std::stringstream ss(line);
        std::string key;
        long value;
        ss >> key >> value;
        if (key == "MemTotal:") {
            mem_total = value;
        } else if (key == "MemAvailable:") {
            mem_available = value;
        }
    }

    if (mem_total > 0 && mem_available > 0) {
        const long mem_used = mem_total - mem_available;
        static constexpr double KB_TO_GB = 1.0 / (1024.0 * 1024.0);
        info.total_gb = mem_total * KB_TO_GB;
        info.used_gb = mem_used * KB_TO_GB;
        info.percentage = static_cast<double>(mem_used) / mem_total * 100.0;
    }
    return info;
}
```

We read the file line-by-line until we find the two rows we're looking for: `"MemTotal"` and `"MemAvailable"`. From those, we can compute the memory used and convert it into gigabytes (since it's usually in kilobytes). Finally we compute the percentage of used RAM and return the struct as info. It's straightforward to incorporate that into the `main` function with some formatting to make it look pretty.

```cpp
// after the CPU usage informatoin
const RamInfo ram_info = get_ram_usage();

std::cout << std::format("{:<12} {:.1f} / {:.1f} GB ({:.1f} %)\n",
                            "RAM Usage:", ram_info.used_gb, ram_info.total_gb,
                            ram_info.percentage);
```

Next up is CPU temperature which is even easier since the corresponding file has only a single value: the CPU temperature (for a given thermal zone) in milli-Celsius.

```cpp
double get_cpu_temperature() {
    const std::filesystem::path path = "/sys/class/thermal/thermal_zone0/temp";

    if (std::filesystem::exists(path)) {
        std::ifstream temp_file(path);
        if (temp_file.is_open()) {
            double temp;
            temp_file >> temp;
            // The value is typically in millidegrees Celsius
            return temp / 1000.0;
        }
    }
    return -1.0;
}
```

In this case, we're using thermal zone 0 but we could pick other thermal zones or print all of them. For the sake of this example, we'll just pick the first one and report it. Incorporating this into the `main` function is even more straightforward.

```cpp
// after the CPU temp info
const double cpu_temp = get_cpu_temperature();

std::cout << std::format("{:<12} {:.1f} °C\n", "CPU Temp:", cpu_temp);
```

Finally we want to print the uptime. Unsurprisingly, this is also in a file `/proc/uptime`!

```cpp
std::string format_uptime() {
    std::ifstream uptime_file("/proc/uptime");
    double uptime_seconds_val = 0.0;
    if (!uptime_file.is_open()) {
        return "N/A";
    }
    uptime_file >> uptime_seconds_val;

    using namespace std::chrono;
    seconds total_seconds(static_cast<long>(uptime_seconds_val));

    const auto d = duration_cast<days>(total_seconds);
    total_seconds -= d;
    const auto h = duration_cast<hours>(total_seconds);
    total_seconds -= h;
    auto m = duration_cast<minutes>(total_seconds);
    total_seconds -= m;
    auto s = total_seconds;

    std::stringstream ss;
    ss << d.count() << "d " << h.count() << "h " << m.count() << "m "
       << s.count() << "s";
    return ss.str();
}
```

The file's first value is the floating-point uptime of the system in seconds which is what we need (the second value is the idle time). We'll read that into seconds and then perform some arithmetic operations to convert it into a nice day, hour, minute, seconds format using the chrono library. Since this is already formatted as a string, printing it in the `main` function is trivial.

```cpp
// after the CPU temp info
const std::string uptime = format_uptime();
std::cout << std::format("{:<12} {}\n", "Uptime:", uptime);
```

Running this, we'll get an output like the following that'll refresh every second.

```
---- Linux System Monitor ----
CPU Usage:   1.5 %
RAM Usage:   5.0 / 15.1 GB (33.1 %)
CPU Temp:    36.0 °C
Uptime:      0d 1h 13m 24s
------------------------------------
```

And that's it! We've completed our MVP of our Linux system monitor. If this were a prototype or some intentional throwaway work, then what we're written is completely acceptable: we have some degree of modularity by abstracting the core logic of the monitor computations in functions and then ordered them into the loop itself (maybe with some block comments acting as separators). In the case of a prototype, intentional throwaway work, or even starting a project in an unfamiliar domain space, over-generalization and over-abstraction tends to slow down progress. In the case of the prototype, it'll be replaced with a better, production version given the learnings from the prototype or scrapped entirely. In the case of intentional throwaway work, the focus is on getting something working as soon as possible so extra work only prolongs the timeline. In the case of working in a novel and unfamiliar domain, it's a bit of a combination of the previous two use-cases in that we don't know what exactly we want and what interfaces are actually important to construct so we want to move fast to get learnings that we can replace with a better system as we learn more.

# Runtime Polymorphic Interface

Now that we have an initial MVP that works, suppose we want to go ahead with this Linux system monitor beyond the prototyping stage so now we have a need to actual generalize. The primary criticism about the current implementation that makes it difficult to generalize is that the computation, printing, and state (for the CPU monitor) are all in the `main` function which can easily get bloated with other kinds of computation, printing, and state from other monitors turning it into a tangled, overlapping mess.

Let's create a class for each monitor. In fact, let's go a step further and define a `virtual` interface that all of the monitors abide by so that the `main` function can just hold a polymorphic `std::vector<std::unique_ptr<Monitor>>` that we can iterate over and ask the monitors to do something uniformly via the `Monitor` interface.

Now we have to sit down and design what that `Monitor` interface looks like especially since we have monitors that operate on heterogeneous data. One candidate might be something like this:

```cpp
class Monitor {
public:
    // don't forget the virtual destructor!
    virtual ~Monitor() = default;

    // fetches the value that the monitor is responsible for
    double get() = 0;

    // returns a formatted string to print
    std::string print(double val) = 0;
};
```

We could even go a step further and generalize the `double` to a generic `T` type like this:

```cpp
template<typename T>
class Monitor {
public:
    virtual ~Monitor() = default;

    // fetches the value that the monitor is responsible for
    T get() = 0;

    // prints the value
    std::string format(const T& val) = 0;
};
```

The monitor classes then might be defined like this:

```cpp
class CpuUsageMonitor : public Monitor<CpuTimes> {
    // ...
};

class RamMonitor : public Monitor<RamInfo> {
    // ...
};

class CpuTempMonitor : public Monitor<double> {
    // ...
};
```

This particular interface is too overly-specific: why does the user need to specify a `get` when there's already a `print` that's going to consume that value anyways? If we were going to do something useful with the value of `get`, then perhaps that interface design has some legitimacy but not in this case. Interfaces should be as minimal as possible to get the job done and no more or less minimal than that. An interface with many similar or unrelated functions usually indicates it needs to be broken up into smaller interface or the inputs and outputs of those functions need to be redesigned. In both cases, we should define the necessary functions on interfaces in a way that only defines what we want from the derived classes and nothing more.

I've seen some people over-design interfaces with many functions with specific inputs and outputs thinking that we need to add *more* required functions and constraints, but, inevitably, there will be derived classes that don't use those required functions that are forced to override those as empty functions just because they're required by the interface. And when new requirements come in, those people think to add more functions to satisfy the new requirements. Rather than adding functions with a high level of specificity, I've found that *removing* specific functions is often more generic. My rationale is that if the interface inhibits derived classes from implementing new requirements on their own, then the interface might be too restrictive already! *Removing* parts of the interface and giving the derived classes *more* freedom and flexibility is the more generic way to go.

All of that being said, interface design is definitely more of an art than a science!

Tying this back to our specific use-case, for our monitors, we don't actually care about the value that's returned, but we just want the monitors to print out what they're monitoring to the screen in whatever way they want. The cleaner interface we'll go with directly captures this requirement:

```cpp
class Monitor {
public:
    virtual ~Monitor() = default;

    // prints to the screen
    virtual void print() = 0;
};
```

We're completely delegating the responsibility of printing the monitored information to the derived classes. There are different alternatives to this interface along the same vein: for example, we could still keep a similar `std::string format()` function and have the `main` function stream `format()` to the output. If we wanted to uniformly log the output of the system monitor, then having that `format` might even be better since we could have the main executor open a log file and write to that. But we're not going to support that use-case for now.

With this new interface, let's move all of the logic of the functions into classes. For example, the (abridged) CPU usage class would look like this:

```cpp
class CpuMonitor : public Monitor {
public:
    void print() override {
        const CpuTimes current_times = read_cpu_times();
        const double cpu_usage
                = calculate_cpu_usage(prev_times_, current_times);
        if (is_first_loop_) {
            std::cout << std::format("{:<12} Calculating...\n", "CPU Usage:");
            is_first_loop_ = false;
        } else {
            std::cout << std::format("{:<12} {:>3.1f} %\n",
                                     "CPU Usage:", cpu_usage);
        }
        prev_times_ = current_times;
    }

private:
    // same as before
    struct CpuTimes {
        //...
    };

    CpuTimes prev_times_{};
    bool is_first_loop_{true};

    // same as before
    CpuTimes read_cpu_times() {
        // ...
    }

    // same as before
    double calculate_cpu_usage(const CpuTimes& prev, const CpuTimes& current) {
        // ...
    }
};

```

The other classes follow similarly (I'll leave them as exercises to the reader 😉). Given these derived classes, we can simplify our `main` function to leverage the new runtime polymorphic interface:

```cpp
int main(int, char**) {
    std::vector<std::unique_ptr<Monitor>> monitors{};
    monitors.emplace_back(std::make_unique<CpuMonitor>());
    monitors.emplace_back(std::make_unique<RamMonitor>());
    monitors.emplace_back(std::make_unique<CpuTempMonitor>());
    monitors.emplace_back(std::make_unique<UptimeMonitor>());

    using namespace std::chrono_literals;

    while (!stop_loop) {
        clear_screen();

        std::cout << "---- Linux System Monitor ----n\n";
        for (const auto& monitor : monitors) {
            monitor->print();
        }
        std::cout << "------------------------------------\n";
        std::cout << "Press Ctrl+C to exit." << std::endl;

        std::this_thread::sleep_for(1s);
    }

    return 0;
}
```

Much simpler since all of the logic is moved out! We're storing all of the monitors polymorphically in a `std::vector<std::unique_ptr<Monitor>>` that we populate once at the start with the derived classes that we later use in the main loop. We're creating concrete instantiations of the derived classes and then storing them into a `std::vector` of the base class so we need the `std::unique_ptr` for runtime polymorphism. Note: when we iterate over `monitors`, we need to use a reference to a `std::unique_ptr` since we can't copy a `std::unique_ptr`.

One very important thing to note that we won't address right now is regarding the memory allocated by the `std::vector<std::unique_ptr<Monitor>>`. Normally, the `std::unique_ptr` will free its memory automatically at the end of the scope, but since we're running an infinite loop, if we use Ctrl-C, we'll raise a `SIGINT` to the process and exit the entire program without cleaning up the memory. This is technically fine since the OS will clean up the memory anyways but it won't be fine for some of the later patterns so we'll address the memory issue when we get there.

Now we're making more progress! Going beyond just functions and directly writing logic into the `main` function, we defined a runtime polymorphic interface and used it to help remove almost all of the business logic from the `main` function. The only monitor-specific code just creates the monitors themselves. This produces the same output as the MVP but is more extensible and maintainable.

# Registry Pattern

One blatant issue with the runtime polymorhpic approach is that, if we want to add a new monitor, we still have to go into the `main` function and add it. This might not be possible if the code that runs the `main` function lives elsewhere where it's not easily modifiable, e.g., the executor takes several hours to build or, in industry, it's owned by a completley different team. Furthermore, this example is simple in that there's just a singular `main` function that we need to modify to add new monitors, but, in real codebases, this place might not be obvious (e.g., the monitor list is in some utility file or other software package somewhere far from the `main` function) or there might be multiple places where the monitor needs to be registered. Hopefully there's documentation on all of the places the module needs to be registered, but, if not, then we'll have to hunt down all of those places!

The **registry pattern** is one technique to invert the place where the derived classes are constructed: rather than constructing them manually in the `main` function's `std::vector`, the idea is to have each monitor *register itself* into a global registry that's stored in the framework code but used by the executor.

As an added step, to decouple monitor registration from monitor construction, we'll store the *factories* that construct the monitors in a global vector and then use the factories to instantiate the monitors into the registry at runtime in the `main` function. Let's first define the factory function and global factory registry.

```cpp
using MonitorFactory = std::function<std::unique_ptr<Monitor>()>;

inline std::vector<MonitorFactory>& getMonitorFactoryRegistry() {
    static std::vector<MonitorFactory> monitor_factories{};
    return monitor_factories;
}
```

Now we need a way for monitors to add themselves to this factory registry. We can take advantage of static initialization and define a dummy static variable that, in its constructor, registers its factory function into the factory registry. For the CPU utilization struct, it might look like this:

```cpp
namespace {
struct RegistrarCpuMonitor {
    RegistrarCpuMonitor() {
        getMonitorFactoryRegistry().push_back([] { return std::make_unique<CpuMonitor>(); });
    }
};
RegistrarCpuMonitor registrarCpuMonitor;
}
```

We're using an unnamed/anonymous namespace so the struct and variable have internal linkage just to prevent it from leaking out of the translation unit/cpp file. In the constructor of our registrar, we access the monitor factory registry and add a factory function/lambda that constructs our `CpuMonitor`. Then we immediately create an instance of it to invoke that constructor as the first thing that happens when the program is executed. (Static initialization happens even before `main` is executed since those variables live in a different segment of the program.)

The logic is the same for each monitor but since we're relying on creating a unique global variable in static storage, there's unfortunately no way to directly write C++ to abstract this away. Our only option is to define a macro that does the same thing.

```cpp
#define REGISTER_MONITOR(MonitorClass) \
    namespace { \
    struct Registrar##MonitorClass { \
        Registrar##MonitorClass() { \
            getMonitorFactoryRegistry().push_back([] { return std::make_unique<MonitorClass>(); }); \
        } \
    }; \
    Registrar##MonitorClass registrar##MonitorClass; \
    }
```

Now we can add this to the end of each monitor class in the global scope.

```cpp
class CpuMonitor : public Monitor {
    // ...
};
REGISTER_MONITOR(CpuMonitor)

class RamMonitor : public Monitor {
    // ...
};
REGISTER_MONITOR(RamMonitor)
```

With all of the monitor factories registered, we need to construct them into a monitor registry that reads the factory registry and invokes the factory functions to construct the monitors.

```cpp
class MonitorRegistry {
public:
    explicit MonitorRegistry(const std::vector<MonitorFactory>& factories) {
        monitors_.reserve(factories.size());
        std::ranges::transform(factories, std::back_inserter(monitors_), [](const auto& factory) {
            return factory();
        });
    }

    ~MonitorRegistry() = default;

    MonitorRegistry(const MonitorRegistry&) = delete;
    MonitorRegistry(MonitorRegistry&&) = delete;

    MonitorRegistry& operator=(const MonitorRegistry&) = delete;
    MonitorRegistry& operator=(MonitorRegistry&&) = delete;

    const std::vector<std::unique_ptr<Monitor>>& getMonitors() const {
        return monitors_;
    }

private:
    std::vector<std::unique_ptr<Monitor>> monitors_{};
};
```

I'm using the new C++20 ranges library to do this but a normal `std::transform` will work as well. Now the `main` function gets simplified even further! We can create a `MonitorRegistry` and iterate over the monitors.

```cpp
int main(int, char**) {
    using namespace std::chrono_literals;

    MonitorRegistry monitor_registry(getMonitorFactoryRegistry());

    while (true) {
        clear_screen();

        std::cout << "---- Linux System Monitor ----n\n";
        for (const auto& monitor : monitor_registry.getMonitors()) {
            monitor->print();
        }
        std::cout << "------------------------------------\n";
        std::cout << "Press Ctrl+C to exit." << std::endl;

        std::this_thread::sleep_for(1s);
    }

    return 0;
}
```

Now the monitors have the responsibility of registering themselves! (One added benefit of this approach is that it's easier to test as well.) With this change, we're moving towards a better architecture where the monitors and main executor can live completely independently to each other, in separate libraries even. At this point, we've hit a good milestone: the monitors are completely independent of each other and the executor.

# Dynamic Plugin Architecture

Going a step further, even the registry pattern assumed that we had the main executor code that we could freely link against at build-time. This might not be the case in some scenarios: in the extreme case, the executor code is proprietary and we don't have access to it. The vendor that supplies it wants to hide their trade secrets so they just provide the framework and obfuscated executable. In another case, perhaps the executor code itself is difficult to directly link against or is too expensive to build each time. In these cases, we'd ideally want to define the plugin in a completely separate shared library and have the executor read and use that library dynamically at runtime.

Fortunately, there's a solution: the dynamic loader. On most systems, shared libraries can be directly loaded into a process's memory space by that process in code. Whenever we create shared library, we can see all of the symbols that the library exports (try running `nm -D <your favorite library>`) and, when we load it into our process, we can get a pointer to any symbol in the library. Suppose we know that a particular symbol referred to a function with a specific signature: we could cast it to a function pointer and invoke it! This is the idea behind plugins in a plugin architecture: we define a function that constructs the monitor, grab a function pointer to it, and use that function pointer to construct the object! Since all of this logic happens at runtime, we could also add functionality to hot-reload any plugins, i.e., re-read the plugin library and re-initialize without ever restarting the executor process!

Let's start by partitioning our classes into separate files and libraries. We'll need a header file for the framework and new export macro that defines the function that will construct a given monitor.

```cpp
// framework.hpp

class Monitor {
public:
    virtual ~Monitor() = default;
    virtual void print() = 0;
};

#define EXPORT_MONITOR(MonitorClass) \
    extern "C" Monitor* createMonitor() { return new MonitorClass{}; }
```

We're using `extern "C"` to ensure the function abides by the C application binary interface (ABI). In the case of C++, for example, that means removing any namespace modifiers. So when we build a monitor library, if we listed the symbols, we'd see a global function called `createMonitor` in the text section of the program! One important note is that since we're using the C ABI, we can't return a proper `std::unique_ptr<Monitor>` (since C has no notion of classes or templates!) so we'll have to freely allocate the monitor using a raw `new`, but, in the executor, we'll immediately wrap it in a `std::unique_ptr<Monitor>` so there won't be any issues with memory. It's always good practice to use resource acquisition is initialization (RAII)/scoped memory management.

Now we can separate each of the monitors into their own separate files and change the macro from `REGISTER_MONITOR` to `EXPORT_MONITOR`.

```cpp
// ram_monitor.cpp
class RamMonitor : public Monitor {
    // ...
};

EXPORT_MONITOR(RamMonitor);
```

Then we can build each of these monitors into their own separate libraries that we'll dynamically load when we run the executor.

Speaking of the executor, we'll modify it yet again to support this. To show how dynamically this plugin architecture works, we're going to put the paths to the plugins into a file and have the executor read that file, load the plugins, and run them. Then we can edit the file with new plugins or remove plugins and re-run the same executor again, without making any code change to it, to see the new monitors on the screen.

Going back to the memory issues with the infinite loop and `SIGINT` were discussing, I'll demonstrate how to get the signal handler set up since it will make a different if we're hot-loading so that we don't keep leaking memory every time we hot-load a plugin. It's pretty simple to set up but uses some C conventions:

```cpp
static std::atomic_bool stop_loop = false;

static void signal_handler(int signum) {
    if (signum == SIGINT) {
        stop_loop = 1;
    }
}
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    // ...
    while (!stop_loop) {
        // ...
    }
}
```

We override the signal handler for Ctrl-C (`SIGINT`) to invoke `signal_handler` which we use cleanly break out of the loop and release any memory or close any resources.

For the dynamic loading, the only three functions we'll need are `dlopen`, `dlsym`, and `dlclose` from C. The first opens the shared library and the last one closes it. The middle one fetches a pointer to a particular symbol. Opening a shared library returns a `void*` handle that must be closed with `dlclose`. One trick we can use is to re-use `std::unique_ptr` but give it a custom "deletor" that doesn't directly delete the `void*`, but just calls `dlclose` on it.

```cpp
struct DlCloser {
    void operator()(void* handle) const {
        if (handle) {
            ::dlclose(handle);
        }
    }
};

// Define a type alias for the unique_ptr with the custom deleter
using DlHandlePtr = std::unique_ptr<void, DlCloser>;

int main(int argc, char* argv[]) {
    // ...
    std::vector<DlHandlePtr> handles{};
    std::vector<std::unique_ptr<Monitor>> monitors{};

    // ...
}
```

Note I'm choosing to use the global scope resolution operator `::` like `::dlclose` since C doesn't have namespaces so all symbols are in the global namespace; using that operator ensures I'm referring to the C version of those functions and not some other namespace free function called `dlclose`. (C functions tend to have really generic names like `open` and `close` so it's possible to conflict with a namespace level `open` and `close` but using `::open` avoids that.) As a reader, it also signifies to me that these are C functions. This is not required, of course, but just something I do to make it easier to read and write. Now we need to do the work of loading each of these handles from the file. We'll make the user provide the text file as an argument to the program.

```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./monitr <path-to-monitors-txt>\n";
        return -1;
    }
    std::signal(SIGINT, signal_handler);

    std::vector<DlHandlePtr> handles{};
    std::vector<std::unique_ptr<Monitor>> monitors{};
    // ...
}
```

We'll read each line of the text file and create a handle from the file path.

```cpp
int main(int argc, char* argv[]) {
    // ...

    std::vector<DlHandlePtr> handles{};
    std::vector<std::unique_ptr<Monitor>> monitors{};

    std::ifstream monitors_file{argv[1]};
    std::string line;
    // read file line-by-line
    while (std::getline(monitors_file, line)) {
        // create handle for the shared library
        DlHandlePtr handle{::dlopen(line.c_str(), RTLD_NOW), DlCloser{}};
        using MonitorCreateFn = Monitor*(*)();
        // get a reference to the createMonitor function and cast it as a function pointer
        MonitorCreateFn monitor_create_fn = (MonitorCreateFn)::dlsym(handle.get(), "createMonitor");
        // invoke the function pointer to create the monitor
        std::unique_ptr<Monitor> monitor{monitor_create_fn()};

        monitors.emplace_back(std::move(monitor));
        handles.emplace_back(std::move(handle));
    }
}
```

The rest of the function is the same as before:

```cpp
int main(int argc, char* argv[]) {
    // ...
    while (!stop_loop) {
        clear_screen();

        std::cout << "---- Linux System Monitor ----n\n";
        for (const auto& monitor : monitors) {
            monitor->print();
        }
        std::cout << "------------------------------------\n";
        std::cout << "Press Ctrl+C to exit." << std::endl;

        std::this_thread::sleep_for(1s);
    }
    return 0;
}
```

Now we have a dynamic plugin architecture! This is one of the most extensible and flexibility kinds of modular architectures. If we edit the text file and re-run the executor, without having to recompile anything, we'll get a different set of monitors printing to the screen!

# Compile-time Polymorphic Interface

Taking a step back from runtime polymorphism, there's another use-case where we already know which monitors we want to run at compile-time. Or perhaps we're running in a very resource-constrained or performance-critical environment where we want to use as many compile-time constructs as we can to help reduce the heap memory allocation or improve performance. Of course, we should measure to verify that this interface is actually a substantial contributing factor to performance!

For runtime polymorphism, we stored the monitors in a `std::vector` but that's a runtime construct that dynamically allocates memory on the heap; furthermore, we have an extra pointer indirection from the virutal function table due to polymorphism which may contribute a tiny bit to performance. A corresponding compile-time construct to a `std::vector` is a `std::tuple`: we define all of the monitors as elements of a `std::tuple` at compile-time in its template parameter pack.

While we don't necessarily have to explicitly enforce the template, we'll swap out our runtime polymorphic interface with a compile-time one enforced on each type in the `std::tuple`. Let's use a C++20 concept!

```cpp
template<typename T>
concept MonitorLike = requires(T a) {
    // requires that any type that abides by MonitorLike<T> must have a
    // function-like object called print that takes no parameters and
    // returns void
    { a.print() } -> std::same_as<void>;
};
```

Rather than directly using a `std::tuple`, we'll create a `MonitorChain` that hides it and calls functions on all of the underlying types. The implementation is short but requires a bit of explanation.

```cpp
// template parameter pack for all of the monitor types
template<typename... Monitors>
    requires(MonitorLike<Monitors> && ...) // apply MonitorLike concept for all types in the chain
class MonitorChain {
public:
    void print() {
        std::apply([](auto&... monitor) {
            // for 3 monitors, expands to (monitor1.print(), monitor2.print(), monitor3.print());
            (monitor.print(), ...);
        }, monitors_);
    }

private:
    // tuple to store our monitors
    std::tuple<Monitors...> monitors_{};
};
```

Using this is straightforward:

```cpp
MonitorChain<CpuMonitor, RamMonitor, CpuTempMonitor, UptimeMonitor>
        monitors{};
// invokes .print() for all above monitors
monitors.print()
```

Unlike runtime polymorphism, we do have to specify all of the types in a way that the compiler can see all of them at compile-time so they all do need to get into the same executable (although you could still split these up into separate software packages and have some headers define them). Using templates, we can achieve a similar kind of polymorphism that has a much greater chance of being almost entirely inlined during compilation.

# Conclusion

In this post, we explored a few different approaches to writing extensible C++ frameworks and architectures. Our motivating example was building a Linux system monitor with sub-monitors that are responsible for reading one specific aspect of the Linux subsystem like CPU usage and uptime. We started with building a functioning application and then looked at different ways to make it more extensible like the registry pattern the plugin architecture. We also explored a way to do the same at compile-time. 

I hope this exploration provides you with some different options on building some truly extensible architectures 🙂
