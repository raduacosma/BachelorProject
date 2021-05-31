#ifndef _INCLUDED_KOLSMIR
#define _INCLUDED_KOLSMIR
#include <tuple>

class KolSmir
{
  public:
    double probability(double z);
    std::tuple<double, double> test(int na, double const *a, int nb, double const *b);
    template <typename T>
    int nint(T x);
};

template <typename T>
inline int KolSmir::nint(T x)
{
    int i;
    if (x >= 0)
    {
        i = int(x + 0.5);
        if (i & 1 && x + 0.5 == T(i))
            i--;
    }
    else
    {
        i = int(x - 0.5);
        if (i & 1 && x - 0.5 == T(i))
            i++;
    }
    return i;
}
#endif
