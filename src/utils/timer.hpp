#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <stack>

namespace tunicorn_goods {

using namespace std;
using namespace std::chrono;

class Timer {
    public:
    // std::stack<clock_t> tictoc_stack;
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic() {
        // tictoc_stack.push(clock());
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    void toc(string msg="") {
        std::cout << msg 
                  << " Time elapsed: "
                  // << ((double)(clock() - tictoc_stack.top())) / (CLOCKS_PER_SEC / 1000)
                  << duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count()
                  << " ms"<<std::endl;
        tictoc_stack.pop();
    }
    void reset(){
      tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

}


#endif