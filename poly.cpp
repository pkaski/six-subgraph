
/* 
 * This source file is part of an experimental software implementation of a
 * delegatable and error-tolerant algorithm for counting of six-vertex 
 * subgraphs in a graph using the "Camelot" framework 
 * (Björklund and Kaski 2016, https://doi.org/10.1145/2933057.2933101 )
 * and (Kaski 2018, https://doi.org/10.1137/1.9781611975055.16 ) 
 *
 * 
 * The source code is subject to the following license.
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2017-2018 P. Kaski
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#include <iostream>
#include <memory>
#include <cstdio>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <unistd.h>
#include <limits.h>
#include <omp.h>

/********************************************************* Basic data types. */

typedef long int   index_t;   // 64-bit signed index type

/********************************* Threshold for paralellizing simple loops. */

index_t par_threshold = (1L << 20);

/******************************************************** Get the host name. */

const char *sysdep_hostname(void)
{
    static char hn[HOST_NAME_MAX];
    gethostname(hn, HOST_NAME_MAX);
    return hn;
}

/***************************************************** Miscellaneous macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }

/********************************** CUDA synchronization and error wrappers. */

#define CUDA_WRAP(err) (error_wrap(err,__FILE__,__LINE__))

#define CUDA_SYNC                                                     \
    cudaDeviceSynchronize();                                          \
    error_wrap(cudaGetLastError(), __FILE__, __LINE__);               \

static void error_wrap(cudaError_t err,
                       const char *fn,
                       int line) {
    if(err != cudaSuccess) {
        std::printf("CUDA error [%s, line %d]: %s\n",
                    fn,
                    line,
                    cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/******************************************************* Timing subroutines. */

#define TIME_STACK_CAPACITY 256

index_t start_stack_top = -1;
double start_stack[TIME_STACK_CAPACITY];

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}


/********************************** Array allocation, deletion, and copying. */

#define MEMTRACK_STACK_CAPACITY 256

struct alloc_track_struct
{
    void *p;
    size_t size;
    struct alloc_track_struct *prev;
    struct alloc_track_struct *next;
};

typedef struct alloc_track_struct alloc_track_t;

index_t alloc_balance = 0;
alloc_track_t alloc_track_root;
size_t alloc_total = 0;

size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

template <typename T>
T *array_allocate(std::size_t n)
{
    if(alloc_balance == 0) {
        alloc_track_root.prev = &alloc_track_root;
        alloc_track_root.next = &alloc_track_root;
    }
    size_t size = sizeof(T)*n;
    T *p = new T[n];
    alloc_balance++;

    alloc_track_t *t = new alloc_track_t;
    t->p = (void *) p;
    t->size = size;
    pnlinknext(&alloc_track_root, t);
    alloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < alloc_total)
            memtrack_stack[i] = alloc_total;

    return p;
}

template <typename T>
void array_delete(const T *p)
{
    alloc_track_t *t = alloc_track_root.next;
    for(;
        t != &alloc_track_root;
        t = t->next) {
        if(t->p == (void *) p)
            break;
    }
    assert(t != &alloc_track_root); // fails on untracked delete
    alloc_total -= t->size;
    pnunlink(t);
    delete t;

    delete[] p;
    alloc_balance--;
}

template <typename T>
struct array_deleter
{
    void operator()(const T *p) { 
        array_delete(p);
    }
};

template <typename T>
void array_copy(index_t n, const T *f, T *g)
{
#pragma omp parallel for if(n >= par_threshold)
    for(index_t i = 0; i < n; i++)
        g[i] = f[i];
}

index_t dev_alloc_balance = 0;
alloc_track_t dev_alloc_track_root;
size_t dev_alloc_total = 0;

size_t dev_memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t dev_memtrack_stack_top = -1;

template <typename T>
T *dev_array_allocate(size_t n)
{
    if(dev_alloc_balance == 0) {
        dev_alloc_track_root.prev = &dev_alloc_track_root;
        dev_alloc_track_root.next = &dev_alloc_track_root;
    }
    size_t size = sizeof(T)*n;
    T *p;

    CUDA_WRAP(cudaMalloc(&p, size));

    dev_alloc_balance++;

    alloc_track_t *t = new alloc_track_t;
    t->p = (void *) p;
    t->size = size;
    pnlinknext(&dev_alloc_track_root, t);
    dev_alloc_total += size;
    for(index_t i = 0; i <= dev_memtrack_stack_top; i++)
        if(dev_memtrack_stack[i] < dev_alloc_total)
            dev_memtrack_stack[i] = dev_alloc_total;

    return p;
}

template <typename T>
void dev_array_delete(T *p)
{
    alloc_track_t *t = dev_alloc_track_root.next;
    for(;  
        t != &dev_alloc_track_root;
        t = t->next) {
        if(t->p == (void *) p)
            break;
    } // caveat: this search can have linear complexity on each delete,
      //         but the most-recently-allocated-first search order 
      //         should be good enough for present purposes
    assert(t != &dev_alloc_track_root); // fails on untracked delete
    dev_alloc_total -= t->size;
    pnunlink(t);
    delete t;

    CUDA_WRAP(cudaFree(p));
    dev_alloc_balance--;
}

template <typename T>
void dev_array_upload(index_t N, const T *p, T *d_p)
{
    CUDA_WRAP(cudaMemcpy(d_p, p, N*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void dev_array_download(index_t N, const T *d_p, T *p)
{
    CUDA_WRAP(cudaMemcpy(p, d_p, N*sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void dev_array_copy(index_t N, const T *d_a, T *d_b)
{
    CUDA_WRAP(cudaMemcpy(d_b, d_a, N*sizeof(T), cudaMemcpyDeviceToDevice));
}

/************************************************ Memory tracking/reporting. */

double inGiB(size_t s) 
{
    return (double) s / (1L << 30);
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = alloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return alloc_total;
}

void dev_push_memtrack(void) 
{
    assert(dev_memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    dev_memtrack_stack[++dev_memtrack_stack_top] = dev_alloc_total;
}

size_t dev_pop_memtrack(void)
{
    assert(dev_memtrack_stack_top >= 0);
    return dev_memtrack_stack[dev_memtrack_stack_top--];    
}

size_t dev_current_mem(void)
{
    return dev_alloc_total;
}

/****************************************************************** Metrics. */

double  gpu_pm_time    = 0.0;
index_t g_pm_stack_top = -1;
double  g_pm_stack[TIME_STACK_CAPACITY];

double  gpu_mm_time    = 0.0;
index_t g_mm_stack_top = -1;
double  g_mm_stack[TIME_STACK_CAPACITY];

bool m_default = true;

void metric_push(bool active)
{
    if(active) {
        push_time();
        push_memtrack();
        dev_push_memtrack();

        assert(g_pm_stack_top + 1 < TIME_STACK_CAPACITY);
        g_pm_stack[++g_pm_stack_top] = gpu_pm_time;
        assert(g_mm_stack_top + 1 < TIME_STACK_CAPACITY);
        g_mm_stack[++g_mm_stack_top] = gpu_mm_time;
    }
}

bool metric_head = false;
double metric_head_time;
double metric_head_g_pm_time;
double metric_head_g_mm_time;
size_t metric_head_peak_mem;
size_t metric_head_dev_peak_mem;

void metric_stop(void)
{
    assert(!metric_head);
    metric_head = true;
    metric_head_time         = pop_time();
    metric_head_peak_mem     = pop_memtrack();
    metric_head_dev_peak_mem = dev_pop_memtrack();

    assert(g_pm_stack_top >= 0);
    metric_head_g_pm_time = gpu_pm_time - g_pm_stack[g_pm_stack_top--];
    assert(g_mm_stack_top >= 0);
    metric_head_g_mm_time = gpu_mm_time - g_mm_stack[g_mm_stack_top--];
}

double metric_time(bool active)
{
    if(active) {
        if(!metric_head)
            metric_stop();
        return metric_head_time;
    } else {
        return 1.0;
    }
}

void metric_pop(bool active, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    if(active) {
        if(!metric_head)
            metric_stop();
        std::vprintf(format, args);
        std::printf(": %13.2fms [%6.2fGiB, %6.2fGiB] %4.2f %4.2f\n", 
                    metric_head_time,
                    inGiB(metric_head_peak_mem), 
                    inGiB(metric_head_dev_peak_mem),
                    metric_head_g_pm_time/metric_head_time,
                    metric_head_g_mm_time/metric_head_time);
        metric_head = false;
    }
    va_end(args);
}

/************************************* List of metrics and their activation. */

bool m_ss_layers            = false;
bool m_gpu_ss_layers        = false;
bool m_gpu_ss               = false;
bool m_quorem               = false;
bool m_quorem_layers        = false;
bool m_quorem_layers_detail = false;

bool m_gpu_scan             = false;
bool m_gpu_transpose        = false;
bool m_gpu_mul              = false;
bool m_gpu_yates            = false;
bool m_gpu_strassen         = false;
bool m_gpu_eval_detail      = false;
bool m_gpu_eval             = true;

/*************************************************************** Miscellany. */

index_t ceil_log2(index_t n)
{
    assert(n > 0);
    index_t k = 0;
    index_t nn = n;
    while(nn > 1) {
        nn = nn/2;
        k++;
    }
    return k + (((1L << k) < n) ? 1 : 0);
}

static index_t index_pow(index_t a, index_t p)
{
    assert(p >= 0);
    if(p == 0)
        return 1;
    return a * index_pow(a, p-1);
}

index_t ceil_logb(index_t b, index_t n)
{
    assert(n > 0);
    index_t k = 0;
    index_t nn = n;
    while(nn > 1) {
        nn = nn/b;
        k++;
    }
    return k + ((index_pow(b,k) < n) ? 1 : 0);
}

void randperm(index_t n, index_t *p)
{
    for(index_t i = 0; i < n; i++)
        p[i] = i;
    for(index_t j = 0; j < n-1; j++) {
        index_t u = j+rand()%(n-j);
        index_t t = p[u];
        p[u] = p[j];
        p[j] = t;
    }
    index_t *f = array_allocate<index_t>(n);
    for(index_t i = 0; i < n; i++)
        f[i] = 0;
    for(index_t i = 0; i < n; i++) {
        index_t u = p[i];
        assert(u >= 0 && u < n);
        f[u]++;
    }
    for(index_t i = 0; i < n; i++)
        assert(f[i] == 1);
    array_delete<index_t>(f);
}

index_t to_interleaved(index_t k, index_t u) 
{
    index_t r = 0;
    for(index_t i = 0; i < 2*k; i++)
        r = r | (((u >> i)&1)<<(2*(i%k)+(i/k)));
    return r;
}

index_t from_interleaved(index_t k, index_t u) 
{
    index_t r = 0;
    for(index_t i = 0; i < 2*k; i++)
        r = r | (((u >> i)&1)<<(k*(i%2)+(i/2)));
    return r;
}

/* Transforms p so that a[p[i]] <= a[p[j]] iff i <= j. */

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

template <typename T>
void heapsort_indirect(index_t n, const T *a, index_t *p)
{
    index_t i;
    index_t x, y, z, t, s;
    
    /* Shift index origin from 0 to 1 */
    p--; 
    /* Build heap */
    for(i = 2; i <= n; i++) {
        x = i;
        while(x > 1) {
            y = PARENT(x);
            if(a[p[x]] <= a[p[y]]) {
                /* heap property ok */
                break;              
            }
            /* Exchange p[x] and p[y] to enforce heap property */
            t = p[x];
            p[x] = p[y];
            p[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(i = n; i > 1; i--) {
        t = p[i];
        /* Delete max */
        p[i] = p[1];
        /* Insert t */
        x = 1;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[p[y]] < a[p[z]]) {
                s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[p[y]] >= a[p[z]] */
            if(a[t] >= a[p[y]]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move p[y] up the heap */
            p[x] = p[y];
            x = y;
        }
        /* Insert here */
        p[x] = t; 
    }
}

/**************************************** Modular arithmetic modulo a prime. */

typedef unsigned int        scalar_t;           // a 32-bit unsigned int
typedef long int            long_signed_t;      // a 64-bit signed int
typedef unsigned long int   long_scalar_t;      // a 64-bit unsigned int

typedef uint4               scalar4_t;          // 4 x scalar_t for CUDA

// Subroutines for working with the prime(s)

template <typename P>
__host__ __device__ inline 
scalar_t mod_mul_montgomery(scalar_t x, scalar_t y);

template <typename P>
__host__ __device__ inline 
scalar_t mod_mul_montgomery_host(scalar_t x, scalar_t y);

template <typename P> class Zp;
template <typename F> int work(int argc, char **argv);

// Include the primes and other batch code here

#include"batch.h" 

// Now continue with the subroutines...

__host__ __device__ inline 
long_signed_t normalize(long_signed_t x, long_signed_t m) 
{
    long_signed_t r = x % m;
    if(r < 0)
        r = m + r; // division truncates towards zero
    return r;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_reduce(long_signed_t x)
{
    return normalize(x, P::modulus);
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_add(scalar_t x, scalar_t y) 
{
    scalar_t r = x+y;
    if(P::modulus >= (1U << 31))
        if(r < x) // x + y overflows beyond scalar_t ?
            r = r + P::overflow_adjust; 
    if(r >= P::modulus) // r overflows modulus ?
        r = r - P::modulus;
    return r;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_neg(scalar_t x) 
{
    return x == 0U ? x : P::modulus - x;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_sub(scalar_t x, scalar_t y) 
{
    return mod_add<P>(x, mod_neg<P>(y));
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_mul_plain(scalar_t x, scalar_t y) 
{
    return ((long_signed_t) x*(long_signed_t) y)%((long_signed_t) P::modulus); 
}

__host__ __device__ inline 
long_signed_t first_bezout_factor(long_signed_t a, long_signed_t b)
{
    long_signed_t old_s = 1;
    long_signed_t old_t = 0;
    long_signed_t old_r = a;
    long_signed_t s = 0;
    long_signed_t t = 1;
    long_signed_t r = b;
    while(r != 0) {
        long_signed_t q = old_r / r;
        long_signed_t save = old_r;
        old_r = r;
        r = save - q*r;
        save = old_s;
        old_s = s;
        s = save - q*s;
        save = old_t;
        old_t = t;
        t = save - q*t;     
    }
    assert(old_r == 1); // fails if a is not a unit mod b
    return old_s; // a*old_s + b*old_t == old_r
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_rand(void)
{
    return mod_reduce<P>((((long_signed_t) rand())<<16)^
                         ((long_signed_t)rand()));
}

template <typename P>
__host__ __device__ inline 
scalar_t montgomery_reduce(long_scalar_t T)
{
    scalar_t m = ((scalar_t) T) * P::montgomery_F;
    scalar_t t = (scalar_t) ((T + ((long_scalar_t) m)*P::modulus) >> 32);
    if(P::modulus > (1U << 31)) {
        // addition above can overflow
        assert(false);
    }
    if(t >= P::modulus)
        t -= P::modulus;
    return t;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_mul_montgomery_host(scalar_t x, scalar_t y)
{
    return montgomery_reduce<P>(((long_scalar_t) x)*y);
}

template <typename P>
__host__ __device__ inline 
scalar_t to_montgomery(scalar_t v)
{
    return mod_mul_montgomery<P>(v, P::montgomery_R2);
}

template <typename P>
__host__ __device__ inline 
scalar_t from_montgomery(scalar_t v)
{
    return mod_mul_montgomery<P>(v, 1U);
}

template <typename P>
__host__ __device__ inline 
scalar_t to_mod(long_scalar_t v)
{
    return to_montgomery<P>(mod_reduce<P>(v));
}

template <typename P>
__host__ __device__ inline 
long_scalar_t from_mod(scalar_t v)
{
    return from_montgomery<P>(v);
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_mul(scalar_t x, scalar_t y)
{
    return mod_mul_montgomery<P>(x, y);
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_zero(void)
{
    return 0U;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_one(void)
{
    return to_mod<P>(1U);
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_two(void)
{
    return to_mod<P>(2U);
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_inv(scalar_t a)
{
    scalar_t a_inv = to_mod<P>(first_bezout_factor(from_mod<P>(a), P::modulus));
    assert(mod_mul<P>(a_inv, a) == mod_one<P>());
    return a_inv;
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_inv2()
{
    return mod_inv<P>(mod_two<P>());
}

template <typename P>
__host__ __device__ inline 
scalar_t mod_inv_2_pow_k(index_t k)
{
    assert(k >= 0);
    scalar_t inv2 = mod_inv2<P>();
    scalar_t r = mod_one<P>();   
    for(index_t j = 0; j < k; j++)
        r = mod_mul<P>(r, inv2);
    return r;
}

/*********************** Scalar class for modular arithmetic modulo a prime. */

template <typename P>
class Zp
{
    scalar_t u;
public:
    static const scalar_t characteristic = P::modulus;
    static const Zp zero, one;

    __host__ __device__ 
    Zp()                : u(mod_zero<P>()) { }
    __host__ __device__ 
    Zp(const Zp &other) : u(other.u)       { }
    __host__ __device__ 
    Zp(scalar_t v)      : u(to_mod<P>(v))  { } 
    __host__ __device__ 
    Zp(scalar_t v, bool no_conv) : u(v)    { } // use with care
    __host__ __device__ 
    ~Zp() { }

    __host__ __device__ 
    scalar_t raw() const       { return u; } // use with care
    __host__ __device__ 
    scalar_t value() const     { return from_mod<P>(u); }
    __host__ __device__ 
    Zp operator-() const       { return Zp(mod_neg<P>(u), true); }
    __host__ __device__ 
    Zp inv(void) const         { return Zp(mod_inv<P>(u), true); }

    __host__ __device__ 
    Zp& operator= (const Zp r) { u = r.u; return *this; }
    __host__ __device__ 
    Zp& operator+=(const Zp r) { u = mod_add<P>(u, r.u); return *this; }
    __host__ __device__ 
    Zp& operator-=(const Zp r) { u = mod_sub<P>(u, r.u); return *this; }
    __host__ __device__ 
    Zp& operator*=(const Zp r) { u = mod_mul<P>(u, r.u); return *this; }
    __host__ __device__ 
    Zp& operator/=(const Zp r) { u = mod_mul<P>(u, mod_inv<P>(r.u));
                                 return *this; }

    __host__ __device__ 
    friend Zp operator+(Zp l, const Zp& r) { l += r; return l; }   
    __host__ __device__ 
    friend Zp operator-(Zp l, const Zp& r) { l -= r; return l; }   
    __host__ __device__ 
    friend Zp operator*(Zp l, const Zp& r) { l *= r; return l; }
    __host__ __device__ 
    friend Zp operator/(Zp l, const Zp& r) { l /= r; return l; }
    
    __host__ __device__ 
    friend bool operator==(const Zp& l, const Zp& r) { return l.u == r.u; }
    __host__ __device__ 
    friend bool operator!=(const Zp& l, const Zp& r) { return !(l == r); }
    __host__ __device__ 
    friend bool operator<(const Zp& l, const Zp& r) { return l.u < r.u; }
    __host__ __device__ 
    friend bool operator<=(const Zp& l, const Zp& r) { return l.u <= r.u; }
    __host__ __device__ 
    friend bool operator>(const Zp& l, const Zp& r) { return !(l <= r); }
    __host__ __device__ 
    friend bool operator>=(const Zp& l, const Zp& r) { return !(l < r); }

    static Zp rand(void) { return mod_rand<P>(); }
    static Zp inv_2_pow_k(index_t k) { return Zp(mod_inv_2_pow_k<P>(k), true); }
};

template <typename P> const Zp<P> Zp<P>::zero = Zp<P>(0U);
template <typename P> const Zp<P> Zp<P>::one  = Zp<P>(1U);

template <typename P>
std::ostream& operator<<(std::ostream& out, const Zp<P>& s)
{
   return out << s.value();
}


/**************************** Basic subroutines for scalar array arithmetic. */

template <typename F>
void array_zero(index_t n, F *f)
{
#pragma omp parallel for if(n >= par_threshold)
    for(index_t i = 0; i < n; i++)
        f[i] = F::zero;
}

template <typename F>
void array_rand(index_t n, F *f)
{
    for(index_t i = 0; i < n; i++)
        f[i] = F::rand();
}

template <typename F>
void array_add(index_t nf, const F *f, 
               index_t ng, const F *g, 
               index_t nh, F *h)
{
    assert(nh >= nf && nh >= ng);
    if(nh != nf || nh != ng) {
#pragma omp parallel for if(nh >= par_threshold)
        for(index_t i = 0; i < nh; i++)
            h[i] = (i < nf ? f[i] : F::zero) + (i < ng ? g[i] : F::zero);
    } else {
#pragma omp parallel for if(nh >= par_threshold)
        for(index_t i = 0; i < nh; i++)
            h[i] = f[i] + g[i];
    }
}

template <typename F>
void array_sub(index_t nf, const F *f, 
               index_t ng, const F *g, 
               index_t nh, F *h)
{
    assert(nh >= nf && nh >= ng);
    if(nh != nf || nh != ng) {
#pragma omp parallel for if(nh >= par_threshold)
        for(index_t i = 0; i < nh; i++)
            h[i] = (i < nf ? f[i] : F::zero) - (i < ng ? g[i] : F::zero);
    } else {
#pragma omp parallel for if(nh >= par_threshold)
        for(index_t i = 0; i < nh; i++)
            h[i] = f[i] - g[i];
    }
}

template <typename F>
void array_neg(index_t nf, const F *f, 
               index_t ng, F *g)
{
    assert(ng == nf);
#pragma omp parallel for if(nf >= par_threshold)
    for(index_t i = 0; i < nf; i++)
        g[i] = -f[i];
}

template <typename F>
void array_scalar_mul(index_t nf, const F *f, 
                      const F s, 
                      index_t nh, F *h)
{
    assert(nf == nh);
#pragma omp parallel for if(nf >= par_threshold)
    for(index_t i = 0; i < nf; i++)
        h[i] = f[i]*s;
}

template <typename F>
void array_convolve(index_t nf, const F *f, 
                    index_t ng, const F *g, 
                    index_t nh, F *h)
{
    assert(nh-1 >= nf-1+ng-1);
#pragma omp parallel for if(nh >= par_threshold)
    for(index_t i = 0; i < nh; i++) {
        F t(F::zero);
        for(index_t j = 0; j <= i; j++)
            t = t + (j < nf ? f[j] : F::zero)*(i-j < ng ? g[i-j] : F::zero);
        h[i] = t;
    }
}

template <typename F>
F array_poly_eval(index_t n, const F *f, const F u)
{
    assert(n >= 1);
    F v = f[n-1];
    for(index_t i = n-2; i >= 0; i--)
        v = v*u + f[i];
    return v;
}

template <typename F>
bool array_eq(index_t nf, const F *f, const F *g)
{
    for(index_t j = 0; j < nf; j++) 
        if(f[j] != g[j])
            return false;    
    return true;     
}

template <typename F>
bool array_poly_eq(index_t nf, const F *f,
                   index_t ng, const F *g)
{
    if(nf > ng) {
        index_t ti = nf;
        nf = ng;
        ng = ti;
        const F *tp = f;
        f = g;
        g = tp;
    }
    for(index_t j = 0; j < ng; j++) {
        if(j >= nf) {
            if(g[j] != F::zero)
                return false;
        } else {
            if(f[j] != g[j])
                return false;
        }
    }
    return true;
}

template <typename F>
index_t array_poly_deg(index_t n, const F *f)
{
    index_t d;
    for(d = n-1; d >= 0; d--)
        if(f[d] != F::zero)
            break;
    return d; // the degree of an identically zero array is -1
}

/* Gather N-length to n-length at 
 * offset, offset+1, ..., offset+n-1 over the D-bundle. */

template <typename F>
void array_gather(index_t D, index_t N, index_t n, 
                  const F *in, F *out, 
                  index_t offset = 0)
{
    assert(N >= n);
#pragma omp parallel for if(D*n >= par_threshold)
    for(index_t v = 0; v < D*n; v++) {
        // in a CUDA kernel need to check v < D*n here 
        index_t v_lo = v % n;
        index_t v_hi = v / n;
        index_t u = v_hi*N + v_lo + offset;
        out[v] = in[u];
    }
}

/* Power-of-two version of the above. */

template <typename F>
void array_gather2(index_t d, index_t N, index_t n, 
                   const F *in, F *out, 
                   index_t offset = -1)
{
    array_gather(1L << d, 1L << N, 1L << n, in, out, 
                 offset < 0 ? 0L : 1L << offset);
}

/* Scatter n-length to N-length with zero-padding 
 * at offset, offset+1, ..., offset+n-1 over the D-bundle. */

template <typename F>
void array_scatter(index_t D, index_t n, index_t N, 
                   const F *in, F *out, 
                   index_t offset = 0)
{
    assert(N >= n);
#pragma omp parallel for if(D*N >= par_threshold)
    for(index_t v = 0; v < D*N; v++) {
        // in a CUDA kernel need to check v < D*N here 
        index_t v_lo = v % N;
        index_t v_hi = v / N;
        index_t u_lo = v_lo - offset;       
        index_t u = v_hi*n + u_lo;
        out[v] = (u_lo >= 0 && u_lo < n) ? in[u] : F::zero;
    }
}

/* Power-of-two version of the above. */

template <typename F>
void array_scatter2(index_t d, index_t n, index_t N,
                    const F *in, F *out,
                    index_t offset = -1)
{
    array_scatter(1L << d, 1L << n, 1L << N, in, out, 
                  offset < 0 ? 0L : 1L << offset);
}



/* Assign ones to components of a power-of-two bundle. */

template <typename F>
void array_monic2(index_t d, index_t n, index_t l, F *f)
{
    assert(l < n && l >= 0);
    for(index_t i = 0; i < (1L << d); i++)
        f[i*(1L << n) + (1L << l)] = F::one;
}

/* Assign each component of a power-of-two bundle to the polynomial 1. */

template <typename F>
void array_one2(index_t d, index_t n, F *out)
{
#pragma omp parallel for if((1L << (d+n)) >= par_threshold)
    for(index_t v = 0; v < (1L << (d+n)); v++)
        out[v] = ((v & ((1L << n)-1)) == 0) ? F::one : F::zero;
}

/* Add one to each component of a power-of-two bundle. */

template <typename F>
void array_add_one2(index_t d, index_t n, F *f) 
{   
    for(index_t i = 0; i < (1L << d); i++)
        f[i*(1L << n)] = f[i*(1L << n)] + F::one;
}


/* Interleave two N-length D-bundles. */

template <typename F>
void array_interleave(index_t D, index_t N, 
                      const F *in0, const F *in1, F *out)
{
#pragma omp parallel for if(2*D*N >= par_threshold)
    for(index_t v = 0; v < 2*D*N; v++) {
        // in a CUDA kernel need to check v < 2*D*N here 
        index_t v_lo = v % N;
        index_t v_hi = v / N;
        index_t v_sel = v % 2;
        v_hi = v_hi/2;        
        index_t u = v_hi*N + v_lo;
        out[v] = v_sel == 0 ? in0[u] : in1[u];
    }
}

/* Reverse components of the N-length D-bundle. */

template <typename F>
void array_rev(index_t D, index_t N, const F *f, F *g)
{
    for(index_t j = 0; j < D; j++)
        for(index_t i = 0; i < N; i++)
            g[j*N+N-1-i] = f[j*N+i];
}


/*****************************************************************************/
/******************** Schoenhage--Strassen multiplication in F[x] / <x^N+1>. */
/*****************************************************************************/

/* Assumes 2 has a multiplicative inverse in F. */

/* 
 * Conventions with parameters:
 * 
 * D = 2^{d} = number of operands 
 * N = 2^{n} = size of operand
 * T = 2^{t} = 2^{ceil(n/2)} = number of limbs in operand
 * M = 2^{m} = 2^{floor(n/2)} = number of F-scalars in limb
 *
 * Observe that N == T*M and, equivalently, n==t+m.
 *
 * Description:
 *
 * To execute one multiplication in the quotient ring F[x] / <x^{N}+1>, 
 * the algorithm will recurse with 2T multiplications in F[x] / <x^{2M}+1>. 
 *
 * The recursion is enabled by a 2T-point FFT using a primitive root
 * of unity of degree 2T in F[x] / <x^{2M}+1>. This primitive root
 * w is 
 *
 *    w = x   (mod x^{2M}+1)  if T = 2M 
 *
 * and 
 * 
 *    w = x^2 (mod x^{2M}+1)  if T = M.
 * 
 * The FFT is implemented as a Cooley-Tukey decimation-in-frequency radix-2 FFT
 * that outputs in bit-reversed (transposed) order of the indices 
 * relative to the natural (input) order. 
 * 
 */

#define SIZE(a,b,c) (1L << ((a)+(b)+(c)))
#define DIGIT2(u,a,b,c) ((u) >> (b+c))
#define DIGIT1(u,a,b,c) (((u) >> (c))&((1L << (b))-1))
#define DIGIT0(u,a,b,c) ((u)&((1L << (c))-1))
#define BUILD(x,y,z,a,b,c) ((x) << ((b)+(c)))+((y) << (c))+(z)

/* Expand from (d,t,m) to (d,t+1,m+1). */

template <typename F>
void ss_expand(index_t d, index_t t, index_t m, const F *in, F *out)
{   
    metric_push(m_ss_layers);
#pragma omp parallel for if(SIZE(d,t+1,m+1) >= par_threshold)
    for(index_t v = 0; v < SIZE(d,t+1,m+1); v++) { 
        index_t i = DIGIT2(v,d,t+1,m+1);
        index_t j = DIGIT1(v,d,t+1,m+1);
        index_t k = DIGIT0(v,d,t+1,m+1);
        index_t u = BUILD(i,j,k,d,t,m);
        out[v] = ((j >= (1L << t)) || (k >= (1L << m))) ? F::zero : in[u];
    }
    double time = metric_time(m_ss_layers);
    double trans_bytes = 5*sizeof(F)*(1L << (d+t+m));   
    metric_pop(m_ss_layers,
               "expand:   "
               "d = %2ld, t = %2ld, m = %2ld         (%6.2lfGiB/s)", 
               d, t, m, trans_bytes/((double)(1L << 30))/(time/1000.0));
}

/* Compress from (l,t+1,m+1) to (l,t,m) with x^{2M} = -1. */

template <typename F>
void ss_compress(index_t d, index_t t, index_t m, const F *in, F *out)
{   
    metric_push(m_ss_layers);
#pragma omp parallel for if(SIZE(d,t,m) >= par_threshold)
    for(index_t v = 0; v < SIZE(d,t,m); v++) {
        index_t M = 1L << m;
        index_t T = 1L << t;
        index_t i = DIGIT2(v,d,t,m);
        index_t j = DIGIT1(v,d,t,m);
        index_t k = DIGIT0(v,d,t,m);
        index_t j_minus_1 = (j-1)&((1L << (t+1))-1);
        index_t u0 = BUILD(i, j,         k,   d,t+1,m+1);
        index_t u1 = BUILD(i, j_minus_1, k+M, d,t+1,m+1);
        index_t u2 = BUILD(i, j+T,       k,   d,t+1,m+1);
        index_t u3 = BUILD(i, j+T-1,     k+M, d,t+1,m+1);
        F t0 = in[u0] + in[u1];
        F t1 = in[u2] + in[u3];
        t0 = t0 - t1;
        out[v] = t0;
    }
    double time = metric_time(m_ss_layers);
    double trans_bytes = 5*sizeof(F)*(1L << (d+t+m));
    metric_pop(m_ss_layers,
               "compress: "
               "d = %2ld, t = %2ld, m = %2ld         (%6.2lfGiB/s)", 
               d, t, m, trans_bytes/((double)(1L << 30))/(time/1000.0));
}

/* Base-case (d,m+1) multiply in F[x] / <x^{2M}+1>. */

template <typename F>
void ss_base_mul(index_t d, index_t m, const F *x, const F *y, F *z)
{   
    metric_push(m_ss_layers);
#pragma omp parallel for if((1L << (d+m+1)) >= par_threshold)
    for(index_t v = 0; v < (1L << (d+m+1)); v++) {
        index_t L = 1L << (m+1);
        F s = F::zero;
        index_t u_base = v & ~(L-1);
        index_t k = v & (L-1);
        for(index_t a = 0; a < L; a++) {
            index_t u0 = u_base + a;
            index_t u1 = u_base + ((k-a)&(L-1));
            F t0 = x[u0];
            F t1 = y[u1];
            F p = t0*t1;
            index_t negate = (k < a) && (a <= k + L);
            if(negate)
                p = -p;
            s = s + p;
        }
        z[v] = s;
    }
    double time = metric_time(m_ss_layers);
    index_t mul_count = 1L << (d+2*(m+1));
    metric_pop(m_ss_layers, 
               "base:     "
               "d = %2ld, n = %2ld                (%6.2lfGmul/s)", 
                d, m+1, mul_count/1e9/(time/1000.0));
}


/* Cooley-Tukey decimation-in-frequency 2T-point FFT over F[x] / <x^{2M}+1>. */

/* Forward (d,t+1,m+1) butterfly at level w = 0,1,...,t. */

template <typename F>
void ss_butterfly_forward(index_t d, index_t t, index_t m, 
                          index_t w, 
                          const F *in, F *out)
{
#pragma omp parallel for if(SIZE(d,t+1,m+1) >= par_threshold)
    for(index_t v = 0; v < SIZE(d,t+1,m+1); v++) {
        index_t s = (t == m) ? 1 : 0;
            // primitive 2T'th root of unity is x^{2^s}
        index_t odd = (v >> (t-w+m+1))&1; 
            // even or odd level-w output of the butterfly?
        index_t jp = (v >> (m+1))&((1L << (t-w))-1);
            // index 0,1,...,T/W-1 for signed shift
        index_t shift = jp << (s+w);
            // actual shift 0,1,...,2T-1 for odd output
        index_t k = DIGIT0(v,d,t+1,m+1);
            // extract coordinate to be shifted
        index_t k_shifted = (k-shift)&((1L << (m+1))-1);
            // do shift with cyclic 2M wrap
        index_t negate = odd && (k < shift) && (shift <= k + (1L << (m+1)));
            // negate odd output if shift did wrap around with x^{2M} == -1
        index_t u_same = (v & ~((1L << (m+1))-1)) + (odd ? k_shifted : k);
            // index to input with same parity (even or odd) as v
        index_t u_opp  = u_same ^ (1L << (t-w+m+1));
            // index to input with opposite parity (even or odd) as v
        F t_same = in[u_same];
            // read same-parity input
        F t_opp = in[u_opp];
            // read opposite-parity input
        if(odd)
            t_same = -t_same;
            // negate same-parity input if odd parity
        F t_sum = t_same + t_opp;
            // sum inputs
        if(negate)
            t_sum = -t_sum;
            // negate output if odd parity and wrapped around with x^{2M} == -1
        out[v] = t_sum;
    }
}

/* Inverse (d,t+1,m+1) butterfly at level w = 0,1,...,t. */

template <typename F>
void ss_butterfly_inverse(index_t d, index_t t, index_t m, 
                          index_t w, 
                          const F *in, F *out)
{
#pragma omp parallel for if(SIZE(d,t+1,m+1) >= par_threshold)
    for(index_t v = 0; v < SIZE(d,t+1,m+1); v++) {
        index_t s = (t == m) ? 1 : 0;
            // primitive 2T'th root of unity is x^{2^s}
        index_t odd = (v >> (t-w+m+1))&1; 
            // even or odd level-w output of the butterfly?
        index_t jp = (v >> (m+1))&((1L << (t-w))-1);
            // index 0,1,...,T/W-1 for signed shift
        index_t shift = jp << (s+w);
            // actual shift 0,1,...,2T-1 for odd output
        index_t k = DIGIT0(v,d,t+1,m+1);
            // extract coordinate to be shifted
        index_t k_shifted = (k+shift)&((1L << (m+1))-1);
            // do shift with cyclic 2M wrap
        index_t negate = (k_shifted < shift) && 
                         (shift <= k_shifted + (1L << (m+1)));
            // negate odd input if shift did wrap around with x^{2M} == -1
        index_t u_base = (v & ~((1L << (m+1))-1));
        index_t flip = (1L << (t-w+m+1));
        index_t u_even_base = u_base ^ (odd ? flip : 0L);
        index_t u_odd_base  = u_even_base ^ flip;
        index_t u_even = u_even_base + k;
        index_t u_odd  = u_odd_base + k_shifted;
        F t_even = in[u_even];
        F t_odd  = in[u_odd];
        if(negate^odd)
            t_odd = -t_odd;
        F t_sum = t_even + t_odd;
        out[v] = t_sum;
    }
}

/* Forward (d,t+1,m+1) FFT. Output is in bit-reversed order. */

template <typename F>
F *ss_fft_forward(index_t d, index_t t, index_t m, 
                  F *in, F *scratch)
{
    for(index_t w = 0; w <= t; w++) {
        metric_push(m_ss_layers);
        ss_butterfly_forward(d, t, m, w, in, scratch);
        double time = metric_time(m_ss_layers);
        double trans_bytes = 3*sizeof(F)*(1L << (d+t+1+m+1));
        metric_pop(m_ss_layers,
                   "forward:  "
                   "d = %2ld, t = %2ld, m = %2ld, w = %2ld (%6.2lfGiB/s)", 
                    d, t, m, w, trans_bytes/((double)(1L << 30))/(time/1000.0));
        F *temp = in;
        in = scratch;
        scratch = temp;
    }
    return in;
}

/* Inverse (d,t+1,m+1) FFT. Assumes input is in bit-reversed order. */

template <typename F>
F *ss_fft_inverse(index_t d, index_t t, index_t m, 
                  F *in, F *scratch)
{
    F z = F::inv_2_pow_k(t+1);
    array_scalar_mul(1L << (d+t+1+m+1), in,
                     z,
                     1L << (d+t+1+m+1), in);
    for(index_t w = t; w >= 0; w--) {
        metric_push(m_ss_layers);
        ss_butterfly_inverse(d, t, m, w, in, scratch);
        double time = metric_time(m_ss_layers);
        double trans_bytes = 3*sizeof(F)*(1L << (d+t+1+m+1));
        metric_pop(m_ss_layers,
                   "inverse:  "
                   "d = %2ld, t = %2ld, m = %2ld, w = %2ld (%6.2lfGiB/s)", 
                    d, t, m, w, trans_bytes/((double)(1L << 30))/(time/1000.0));
        F *temp = in;
        in = scratch;
        scratch = temp;
    }
    return in;
}

/* Recursive (d,n),(d,n) -> (d,n) multiplication in F[x]/<x^N+1>. */

template <typename F>
void host_ss_mul(index_t d, index_t n, const F *x, const F *y, F *z)
{
    assert(n >= 0);

    if(n <= 10) { // was 10, was 12
        bool have_buf = false;
        F *out = z;
        if(x == z || y == z) {
            have_buf = true;
            out = array_allocate<F>(1L << (d+n));
        }            
        if(n == 0)
            out[0] = x[0]*y[0];
        else
            ss_base_mul(d, n-1, x, y, out);
        if(have_buf) {
            array_copy(1L << (d+n), out, z);
            array_delete(out);
        }
        return;
    }

    index_t m = n/2;
    index_t t = n-m;
    assert(m <= t);

    F *xe  = array_allocate<F>(1L << (d+t+1+m+1));
    F *xes = array_allocate<F>(1L << (d+t+1+m+1));
    ss_expand(d, t, m, x, xe);
    F *xef = ss_fft_forward(d, t, m, xe, xes);
    if(xef == xe)
        array_delete(xes);
    else
        array_delete(xe);

    F *ye  = array_allocate<F>(1L << (d+t+1+m+1));
    F *yes = array_allocate<F>(1L << (d+t+1+m+1));
    ss_expand(d, t, m, y, ye);
    F *yef = ss_fft_forward(d, t, m, ye, yes);
    if(yef == ye)
        array_delete(yes);
    else
        array_delete(ye);

    F *zef = array_allocate<F>(1L << (d+t+1+m+1));
    index_t calls = d+t+1;
    index_t block = 0;
    while(block + m + 1 < 25 && calls > 0) {
        block++;
        calls--;
    }

    F *xp = xef;
    F *yp = yef;
    F *zp = zef;
    for(index_t c = 0; c < (1L << calls); c++) {
        ss_mul(block, m+1, xp, yp, zp);
//        host_ss_mul(block, m+1, xp, yp, zp);
        xp = xp + (1L << (block + m + 1));
        yp = yp + (1L << (block + m + 1));
        zp = zp + (1L << (block + m + 1));      
    }

    array_delete(yef);
    array_delete(xef);

    F *zes = array_allocate<F>(1L << (d+t+1+m+1));
    F *ze = ss_fft_inverse(d, t, m, zef, zes);

    if(ze == zef)
        array_delete(zes);
    else
        array_delete(zef);

    ss_compress(d, t, m, ze, z);

    array_delete(ze);
}


/********************************************************************* CUDA. */

/* Expand from (d,t,m) to (d,t+1,m+1). */

template <typename F>
__global__
void ker_expand(index_t d, index_t t, index_t m, 
                F *d_in, F *d_out)
{   
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t i = DIGIT2(v,d,t+1,m+1);
    index_t j = DIGIT1(v,d,t+1,m+1);
    index_t k = DIGIT0(v,d,t+1,m+1);
    index_t u = BUILD(i,j,k,d,t,m);
    d_out[v] = ((j >= (1L << t)) || (k >= (1L << m))) ? F(0U) : d_in[u];
}

template <typename F>
void dev_expand(index_t d, index_t t, index_t m, 
                F *d_in, F *d_out)
{   
    metric_push(m_gpu_ss_layers);
    index_t dg = d+t+1+m+1 - 5; assert(dg >= 0);
    index_t dgx = dg >= 16 ? 15 : dg;
    index_t dgy = dg >= 16 ? dg - 15 : 0;
    dim3 dg2(1 << dgx, 1 << dgy);
    dim3 db2(1 << 5, 1);
    ker_expand<<<dg2,db2>>>(d, t, m, d_in, d_out); 
    CUDA_SYNC;
    double time = metric_time(m_gpu_ss_layers);
    double trans_bytes = 5*sizeof(F)*(1L << (d+t+m));
    metric_pop(m_gpu_ss_layers,
               "dev_expand:   "
               "d = %2ld, t = %2ld, m = %2ld         (%6.2lfGiB/s)", 
               d, t, m, trans_bytes/((double)(1L << 30))/(time/1000.0));
}

/* Compress from (l,t+1,m+1) to (l,t,m) with x^{2M} = -1. */

template <typename F>
__global__
void ker_compress(index_t d, index_t t, index_t m, 
                  F *d_in, F *d_out)
{   
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t M = 1L << m;
    index_t T = 1L << t;
    index_t i = DIGIT2(v,d,t,m);
    index_t j = DIGIT1(v,d,t,m);
    index_t k = DIGIT0(v,d,t,m);
    index_t j_minus_1 = (j-1)&((1L << (t+1))-1);
    index_t u0 = BUILD(i, j,         k,   d,t+1,m+1);
    index_t u1 = BUILD(i, j_minus_1, k+M, d,t+1,m+1);
    index_t u2 = BUILD(i, j+T,       k,   d,t+1,m+1);
    index_t u3 = BUILD(i, j+T-1,     k+M, d,t+1,m+1);
    F t0 = d_in[u0] + d_in[u1];
    F t1 = d_in[u2] + d_in[u3];
    t0 = t0 - t1;
    d_out[v] = t0;
}

template <typename F>
void dev_compress(index_t d, index_t t, index_t m, 
                  F *d_in, F *d_out)
{   
    metric_push(m_gpu_ss_layers);
    index_t dg = d+t+m - 5; assert(dg >= 0);
    index_t dgx = dg >= 16 ? 15 : dg;
    index_t dgy = dg >= 16 ? dg - 15 : 0;
    dim3 dg2(1 << dgx, 1 << dgy);
    dim3 db2(1 << 5, 1);
    ker_compress<<<dg2,db2>>>(d, t, m, d_in, d_out);
    CUDA_SYNC;
    double time = metric_time(m_gpu_ss_layers);
    double trans_bytes = 5*sizeof(F)*(1L << (d+t+m));
    metric_pop(m_gpu_ss_layers,
               "dev_compress: "
               "d = %2ld, t = %2ld, m = %2ld         (%6.2lfGiB/s)", 
               d, t, m, trans_bytes/((double)(1L << 30))/(time/1000.0));
}

/* Base-case (d,m+1) multiply in F[x] / <x^{2M}+1>. */

template <typename F>
__global__
void ker_base_mul(index_t d, index_t m, 
                  F *d_x, F *d_y, F *d_z)
{   
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t L = 1L << (m+1);
    F s = F(0U);
    index_t u_base = v & ~(L-1);
    index_t k = v & (L-1);
    for(index_t a = 0; a < L; a++) {
        index_t u0 = u_base + a;
        index_t u1 = u_base + ((k-a)&(L-1));
        F t0 = d_x[u0];
        F t1 = d_y[u1];
        F p = t0 * t1;
        index_t negate = (k < a) && (a <= k + L);
        if(negate)
            p = -p;
        s = s + p;
    }
    d_z[v] = s;
}

template <typename F>
void dev_base_mul(index_t d, index_t m, 
                  F *d_x, F *d_y, F *d_z)
{   
    metric_push(m_gpu_ss_layers);
    index_t dg = d+m+1 - 5; assert(dg >= 0);
    index_t dgx = dg >= 16 ? 15 : dg;
    index_t dgy = dg >= 16 ? dg - 15 : 0;
    dim3 dg2(1 << dgx, 1 << dgy);
    dim3 db2(1 << 5, 1);
    ker_base_mul<<<dg2,db2>>>(d, m, d_x, d_y, d_z);
    CUDA_SYNC;
    double time = metric_time(m_gpu_ss_layers);
    index_t mul_count = 1L << (d+2*(m+1));
    metric_pop(m_gpu_ss_layers, 
               "dev_base:     "
               "d = %2ld, n = %2ld                (%6.2lfGmul/s)", 
                d, m+1, mul_count/1e9/(time/1000.0));
}

/* Cooley-Tukey decimation-in-frequency 2T-point FFT over F[x] / <x^{2M}+1>. */

/* Scalar multiplication kernel. */

template <typename F>
__global__
void ker_poly_scalar_mul(F *d_in, F *d_out, F s)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    d_out[v] = d_in[v] * s;
}

/* Forward (d,t+1,m+1) butterfly at level w = 0,1,...,t. */

template <typename F>
__global__
void ker_butterfly_forward(index_t d, index_t t, index_t m, 
                           index_t w, 
                           F *d_in, F *d_out)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t s = (t == m) ? 1 : 0;
        // primitive 2T'th root of unity is x^{2^s}
    index_t odd = (v >> (t-w+m+1))&1; 
        // even or odd level-w output of the butterfly?
    index_t jp = (v >> (m+1))&((1L << (t-w))-1);
        // index 0,1,...,T/W-1 for signed shift
    index_t shift = jp << (s+w);
        // actual shift 0,1,...,2T-1 for odd output
    index_t k = DIGIT0(v,d,t+1,m+1);
        // extract coordinate to be shifted
    index_t k_shifted = (k-shift)&((1L << (m+1))-1);
        // do shift with cyclic 2M wrap
    index_t negate = odd && (k < shift) && (shift <= k + (1L << (m+1)));
        // negate odd output if shift did wrap around with x^{2M} == -1
    index_t u_same = (v & ~((1L << (m+1))-1)) + (odd ? k_shifted : k);
        // index to input with same parity (even or odd) as v
    index_t u_opp  = u_same ^ (1L << (t-w+m+1));
        // index to input with opposite parity (even or odd) as v
    F t_same = d_in[u_same];
        // read same-parity input
    F t_opp = d_in[u_opp];
        // read opposite-parity input
    if(odd)
        t_same = -t_same;
        // negate same-parity input if odd parity
    F t_sum = t_same + t_opp;
        // sum inputs
    if(negate)
        t_sum = -t_sum;
    // negate output if odd parity and wrapped around with x^{2M} == -1
    d_out[v] = t_sum;
}

/* Inverse (d,t+1,m+1) butterfly at level w = 0,1,...,t. */

template <typename F>
__global__
void ker_butterfly_inverse(index_t d, index_t t, index_t m, 
                           index_t w, 
                           F *d_in, F *d_out)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t s = (t == m) ? 1 : 0;
        // primitive 2T'th root of unity is x^{2^s}
    index_t odd = (v >> (t-w+m+1))&1; 
        // even or odd level-w output of the butterfly?
    index_t jp = (v >> (m+1))&((1L << (t-w))-1);
        // index 0,1,...,T/W-1 for signed shift
    index_t shift = jp << (s+w);
        // actual shift 0,1,...,2T-1 for odd output
    index_t k = DIGIT0(v,d,t+1,m+1);
        // extract coordinate to be shifted
    index_t k_shifted = (k+shift)&((1L << (m+1))-1);
        // do shift with cyclic 2M wrap
    index_t negate = (k_shifted < shift) && 
                         (shift <= k_shifted + (1L << (m+1)));
        // negate odd input if shift did wrap around with x^{2M} == -1
    index_t u_base = (v & ~((1L << (m+1))-1));
    index_t flip = (1L << (t-w+m+1));
    index_t u_even_base = u_base ^ (odd ? flip : 0L);
    index_t u_odd_base  = u_even_base ^ flip;
    index_t u_even = u_even_base + k;
    index_t u_odd  = u_odd_base + k_shifted;
    F t_even = d_in[u_even];
    F t_odd  = d_in[u_odd];
    if(negate^odd) 
        t_odd = -t_odd;
    F t_sum = t_even + t_odd;
    d_out[v] = t_sum;
}

/* Forward (d,t+1,m+1) FFT. Output is in bit-reversed order. */

template <typename F>
F *dev_fft_forward(index_t d, index_t t, index_t m, 
                   F *d_in, F *d_scratch)
{
    for(index_t w = 0; w <= t; w++) {
        metric_push(m_gpu_ss_layers);
        index_t dg = d+t+1+m+1 - 5; assert(dg >= 0);
        index_t dgx = dg >= 16 ? 15 : dg;
        index_t dgy = dg >= 16 ? dg - 15 : 0;
        dim3 dg2(1 << dgx, 1 << dgy);
        dim3 db2(1 << 5, 1);
        ker_butterfly_forward<<<dg2,db2>>>(d, t, m, w, d_in, d_scratch);
        CUDA_SYNC;
        double time = metric_time(m_gpu_ss_layers);
        double trans_bytes = 3*sizeof(F)*(1L << (d+t+1+m+1));
        metric_pop(m_gpu_ss_layers,
                   "dev_forward:  "
                   "d = %2ld, t = %2ld, m = %2ld, w = %2ld (%6.2lfGiB/s)", 
                    d, t, m, w, trans_bytes/((double)(1L << 30))/(time/1000.0));
        F *d_temp = d_in;
        d_in = d_scratch;
        d_scratch = d_temp;
    }

    return d_in;
}

/* Inverse (d,t+1,m+1) FFT. Assumes input is in bit-reversed order. */

template <typename F>
F *dev_fft_inverse(index_t d, index_t t, index_t m, 
                   F *d_in, F *d_scratch)
{
    {
        F z = F::inv_2_pow_k(t+1);

        metric_push(m_gpu_ss_layers);
        index_t dg = d+t+1+m+1 - 5; assert(dg >= 0);
        index_t dgx = dg >= 16 ? 15 : dg;
        index_t dgy = dg >= 16 ? dg - 15 : 0;
        dim3 dg2(1 << dgx, 1 << dgy);
        dim3 db2(1 << 5, 1);
        ker_poly_scalar_mul<<<dg2,db2>>>(d_in, d_scratch, z);
        CUDA_SYNC;
        double time = metric_time(m_gpu_ss_layers);
        double trans_bytes = 2*sizeof(F)*(1L << (d+t+1+m+1));
        metric_pop(m_gpu_ss_layers, 
                   "dev_scalar:   "
                   "d = %2ld, t = %2ld, m = %2ld         (%6.2lfGiB/s)", 
                    d, t, m, trans_bytes/((double)(1L << 30))/(time/1000.0));
        F *d_temp = d_in;
        d_in = d_scratch;
        d_scratch = d_temp;
    }
    
    for(index_t w = t; w >= 0; w--) {

        metric_push(m_gpu_ss_layers);
        index_t dg = d+t+1+m+1 - 5; assert(dg >= 0);
        index_t dgx = dg >= 16 ? 15 : dg;
        index_t dgy = dg >= 16 ? dg - 15 : 0;
        dim3 dg2(1 << dgx, 1 << dgy);
        dim3 db2(1 << 5, 1);
        ker_butterfly_inverse<<<dg2,db2>>>(d, t, m, w, d_in, d_scratch);
        CUDA_SYNC;
        double time = metric_time(m_gpu_ss_layers);
        double trans_bytes = 3*sizeof(F)*(1L << (d+t+1+m+1));
        metric_pop(m_gpu_ss_layers,
                   "dev_inverse:  "
                   "d = %2ld, t = %2ld, m = %2ld, w = %2ld (%6.2lfGiB/s)", 
                    d, t, m, w, trans_bytes/((double)(1L << 30))/(time/1000.0));
        F *d_temp = d_in;
        d_in = d_scratch;
        d_scratch = d_temp;
    }

    return d_in;
}

/* Recursive (d,n),(d,n) -> (d,n) multiplication in F[x]/<x^N+1>. */

template <typename F>
void dev_ss_mul(index_t d, index_t n, 
                F *d_x, F *d_y, F *d_z)
{
    assert(n >= 1);

    if(n <= 4) {
        dev_base_mul(d, n-1, d_x, d_y, d_z);
        return;
    }

    index_t m = n/2;
    index_t t = n-m;

    F *d_xe = dev_array_allocate<F>(1L << (d+t+1+m+1));
    F *d_xes = dev_array_allocate<F>(1L << (d+t+1+m+1));
    dev_expand(d, t, m, d_x, d_xe);
    F *d_xef = dev_fft_forward(d, t, m, d_xe, d_xes);
    if(d_xef == d_xe)
        dev_array_delete(d_xes);
    else
        dev_array_delete(d_xe);

    F *d_ye = dev_array_allocate<F>(1L << (d+t+1+m+1));
    F *d_yes = dev_array_allocate<F>(1L << (d+t+1+m+1));
    dev_expand(d, t, m, d_y, d_ye);
    F *d_yef = dev_fft_forward(d, t, m, d_ye, d_yes);
    if(d_yef == d_ye)
        dev_array_delete(d_yes);
    else
        dev_array_delete(d_ye);

    F *d_zef = dev_array_allocate<F>(1L << (d+t+1+m+1));
    index_t calls = d+t+1;
    index_t block = 0;
    // was 24
    while(block + m + 1 < 24 && calls > 0) {
        block++;
        calls--;
    }
    F *d_xp = d_xef;
    F *d_yp = d_yef;
    F *d_zp = d_zef;
    for(index_t c = 0; c < (1L << calls); c++) {
        dev_ss_mul(block, m+1, d_xp, d_yp, d_zp);
        d_xp = d_xp + (1L << (block + m + 1));
        d_yp = d_yp + (1L << (block + m + 1));
        d_zp = d_zp + (1L << (block + m + 1));      
    }

    dev_array_delete(d_yef);
    dev_array_delete(d_xef);

    F *d_zes = dev_array_allocate<F>(1L << (d+t+1+m+1));
    F *d_ze = dev_fft_inverse(d, t, m, d_zef, d_zes);

    if(d_ze == d_zef)
        dev_array_delete(d_zes);
    else
        dev_array_delete(d_zef);

    dev_compress(d, t, m, d_ze, d_z);

    dev_array_delete(d_ze);

    // xe xes
    // xef
    // xef ye yes
    // xef yef
    // xef yef zef 
    // zes ze
    // ze
    //
    // peak working = 12 * D * N scalars (not including input and output); 
    // input and output = 3 * D * N scalars
    // total = 15 * D * N scalars
    //       = 60 * D * N bytes (with 4-byte scalars)
    //
    //       ~ 6 + d + n (capacity for 33 assuming 8+ GiB glob)

}

/* Entry point for device multiplication. */

template <typename F>
void gpu_ss_mul(index_t d, index_t n, 
                const F *x, const F *y, F *z)
{
    if(n == 0) {
        host_ss_mul(d, n, x, y, z);
        return;
    }   
    
    index_t N = 1L << (d+n);

    F *d_x = dev_array_allocate<F>(N);
    F *d_y = dev_array_allocate<F>(N);
    F *d_z = dev_array_allocate<F>(N);
    dev_array_upload(N, x, d_x);
    dev_array_upload(N, y, d_y);
    metric_push(m_gpu_ss);
    push_time();
    dev_ss_mul(d, n, d_x, d_y, d_z);
    gpu_pm_time += pop_time();
    metric_pop(m_gpu_ss,
               "gpu_ss:    "
               "d = %2ld, n = %2ld, N = %10ld"
               "                                                        ",
               d, n, 1L << N);

    dev_array_download(N, d_z, z);
    dev_array_delete(d_z);
    dev_array_delete(d_y);
    dev_array_delete(d_x);
}

/***************************************************************** End CUDA. */

/* Arbiter between host and device multiplication. */

template <typename F>
void ss_mul(index_t d, index_t n, 
            const F *x, const F *y, F *z)
{
    if(d+n >= 10 && d+n <= 27) { 
        gpu_ss_mul(d, n, x, y, z);
    } else {
        host_ss_mul(d, n, x, y, z);
    }
}


/*****************************************************************************/
/*** Other arithmetic subroutines that rely on multiplication (unoptimized). */
/*****************************************************************************/

/* Subroutines for inverse modulo x^{L}. */

template <typename F>
void array_assign_to2(index_t d, index_t N, index_t k, 
                      const F *in, F *out,
                      const bool negate = false,
                      const F constant_offset = F::zero)
{
    // in  = a (d,N)-vector
    // out = a (d,k+1)-vector
    // out = (constant_offset*x^0 + (-1)^{negate}*in) mod x^{2^k} 
#pragma omp parallel for if((1L << (d+k+1)) >= par_threshold)
    for(index_t v = 0; v < (1L << (d+k+1)); v++) {
        index_t v_lo  = v&((1L << k)-1);
        index_t v_pad = (v >> k)&1;
        index_t v_hi  = v >> (k+1);
        index_t v_constant = (v_lo == 0) && (v_pad == 0);
        index_t u_hi = v_hi * N;
        F s = v_constant ? constant_offset : F::zero;
        F t = (v_pad == 0 && v_lo < N) ? in[u_hi + v_lo] : F::zero;
        out[v] = negate ? s - t : s + t;
    }
}

template <typename F>
void array_assign_from2(index_t d, index_t k, index_t L, F *in, F *out)
{
    // in  = a (d,k+1)-vector
    // out = a (d,L)-vector
#pragma omp parallel for if((1L << d)*L >= par_threshold)
    for(index_t v = 0; v < (1L << d)*L; v++) {
        // need to check v < (1L << d)*L here in a CUDA kernel
        index_t v_lo = v % L;
        index_t v_hi = v / L;
        index_t u = v_hi*(1 << (k+1)) + v_lo; 
        out[v] = in[u];
    }
}

template <typename F>
void array_inverse_mod_x_pow_L(index_t d, index_t N, 
                               const F *f, index_t L, F *g)
{
    // f = a (d,N)-vector
    // g = a (d,L)-vector

    index_t l = ceil_log2(L);
    F *gg = array_allocate<F>(1L << (d+l+1));
    F *t  = array_allocate<F>(1L << (d+l+1));

    array_one2(d, 1, t); 
    index_t k = 0;
    while(k < l) {
        // Invariant: t is a (d,k+1)-array that holds the inverse mod x^{2^k}
        //            in the least 2^k coefficients of each component

        k++;
        // Now make the invariant hold for k, assuming it holds for k-1  ...
        metric_push(m_quorem_layers_detail);
        array_assign_to2(d, 1L << k, k, t, gg); 
            // gg = t                    (as a (d,k+1)-array)
        array_assign_to2(d, N, k, f, t); // could-reverse-assign here
            // t = f mod x^{2^k}         (as a (d,k+1)-array)
        ss_mul(d, k+1, t, gg, t); // could halve the length here to k
            // t = fg                    (as a (d,k+1)-array)
        array_assign_to2(d, 1L << (k+1), k, t, t, 1, F::one + F::one);
            // t = (2-fg) mod x^{2^k}    (as a (d,k+1)-array)
            // Caveat: reading and writing to the same array here       
        ss_mul(d, k+1, t, gg, gg); // need length k+1 due to cyclic wrap
            // g = (2-fg)g               (as a (d,k+1)-array)
        array_assign_to2(d, 1L << (k+1), k, gg, t);
            // t = g mod x^{2^k}         (as a (d,k+1)-array)
        // The invariant now holds
        metric_pop(m_quorem_layers_detail,
                   "invlayer:  "
                   "d = %2ld,                              "
                   "l = %2ld, L = %10ld, N = %10ld, k = %2ld",
                   d, l, L, N, k);
    }
    array_assign_from2(d, k, L, t, g);
    array_delete(t);
    array_delete(gg);
}


/* Batch quotient and remainder for monic divisors. */

template <typename F>
void array_quorem_monic(index_t d,
                        index_t n, const F *a,
                        index_t m, const F *b,
                        F *q,
                        F *r)
{
    assert(n >= 1 && m >= 1);

    if(m > n) {
        for(index_t j = 0; j < (1L << d); j++)
            for(index_t i = 0; i < n; i++)
                r[j*n+i] = a[j*n+i];
        // Caveat: should also set up a zero quotient.
        return;
    }

    /* Invariant: m <= n */
    index_t D = 1L << d;
    for(index_t i = 0; i < D; i++)
        assert(b[i*m+m-1] == F::one); // check that divisors are monic
    index_t k = 1+ceil_log2(n);
    index_t K = 1L << k;
    index_t l = 1+ceil_log2(n-m+1);
    index_t L = 1L << l;

    metric_push(m_quorem);

    F *s = array_allocate<F>(1L << (d+k));
    F *t = array_allocate<F>(1L << (d+k));
    F *u = array_allocate<F>(1L << (d+k));

    // Reverse divisors

    metric_push(m_quorem_layers);
    array_rev(D, m, b, s); 
    metric_pop(m_quorem_layers,
               "reverse:   "
               "d = %2ld, k = %2ld, l = %2ld, "
               "K = %10ld, L = %10ld, "
               "n = %10ld, m = %10ld",
               d, k, l, K, L, n, m);

    // Compute truncated inverse of each reversed divisor (to u)

    metric_push(m_quorem_layers);
    array_inverse_mod_x_pow_L(d, m, s, n-m+1, u);
    metric_pop(m_quorem_layers,
               "inverse:   "
               "d = %2ld, k = %2ld, l = %2ld, "
               "K = %10ld, L = %10ld, "
               "n = %10ld, m = %10ld",
               d, k, l, K, L, n, m);

    // Compute quotients (to q)

    metric_push(m_quorem_layers);
    array_scatter(D, n-m+1, L, u, s);
    array_rev(D, n, a, t);
    array_gather(D, n, n-m+1, t, u);
    array_scatter(D, n-m+1, L, u, t);
    ss_mul(d, l, s, t, u);
    array_gather(D, L, n-m+1, u, t);
    array_rev(D, n-m+1, t, q);
    metric_pop(m_quorem_layers,
               "quotient:  "
               "d = %2ld, k = %2ld, l = %2ld, "
               "K = %10ld, L = %10ld, "
               "n = %10ld, m = %10ld",
               d, k, l, K, L, n, m);

    // Compute remainders (to r)

    metric_push(m_quorem_layers);
    array_scatter(D, n-m+1, K, q, t);
    array_scatter(D, m, K, b, s);
    ss_mul(d, k, s, t, s); // s = qb
    array_gather(D, K, m-1, s, t);
    array_gather(D, n, m-1, a, u);
    array_sub(D*(m-1), u, D*(m-1), t, D*(m-1), r);
    metric_pop(m_quorem_layers,
               "remainder: "
               "d = %2ld, k = %2ld, l = %2ld, "
               "K = %10ld, L = %10ld, "
               "n = %10ld, m = %10ld",
               d, k, l, K, L, n, m);

    metric_pop(m_quorem,
               "quorem:    "
               "d = %2ld, k = %2ld, l = %2ld, "
               "K = %10ld, L = %10ld, "
               "n = %10ld, m = %10ld",
               d, k, l, K, L, n, m);

    array_delete(u);
    array_delete(t);
    array_delete(s);
}

/* Non-monic quotient and remainder. */

template <typename F>
void array_poly_quorem(index_t n, const F *a,
                       index_t m, const F *b,
                       F *q, 
                       F *r)
{
    F t = b[m-1];
    F s = t.inv();

    F *b_monic = array_allocate<F>(m);
    array_scalar_mul(m, b, s, m, b_monic);
    array_quorem_monic(0, n, a, m, b_monic, q, r);
    array_scalar_mul(n-m+1, q, s, n-m+1, q);
    array_delete(b_monic);
}


/********************** Batch evaluation and interpolation with subproducts. */

/* Batch evaluation at p points with the subproduct algorithm. */

template <typename F>
void array_poly_batch_eval(index_t n, const F *f, 
                           index_t p, const F *u, 
                           F *f_u) 
{
    assert(p >= 1);
    index_t k = ceil_log2(p); // Evaluate at 2^k points
    index_t K = 1L << k;
    
    /* Set up a subproduct tree with 2^{k+1}-1 nodes. */

    /* Level l = 0,1,...,k has 2^l nodes, each of degree 2^{k-l}.
     * Since the polynomials are monic, the tree will omit
     * the leading 1 from each polynomial. */

    F *a = array_allocate<F>(1L << (k+1));
    F *b = array_allocate<F>(1L << (k+1));
    F *s = array_allocate<F>(1L << (k+1));
    F *t = array_allocate<F>(1L << (k+1));
    F *q = array_allocate<F>(1L << (k+1));
    
    F *sub = array_allocate<F>((k+1)*K); 
      // (k+1)*2^k scalars, K = 2^k scalars at each level

    for(index_t i = 0; i < K; i++)
        sub[k*K+i] = i < p ? -u[i] : F::zero;

    /* Process the internal nodes one level at a time from bottom to top. */
    for(index_t l = k-1; l >= 0; l--) {
        
        /* For i = 0,...,1L << l, multiply nodes 2*i and 2*i+1 at level l+1. */

        array_gather2 (l, k-l, k-(l+1), sub + (l+1)*K, q);
        array_scatter2(l, k-(l+1), k-l, q, s);
        array_gather2 (l, k-l, k-(l+1), sub + (l+1)*K, q, k-(l+1));
        array_scatter2(l, k-(l+1), k-l, q, t);

        array_monic2(l, k-l, k-(l+1), s);
        array_monic2(l, k-l, k-(l+1), t);

        ss_mul(l, k-l, s, t, sub + l*K); //  + i*(1L << (k-l))

        array_add_one2(l, k-l, sub + l*K); // cancel monic -1 wrap by adding 1
    }

    /* Descend down the subproduct tree and compute remainders. */

    /* Set up the root divisor & compute root remainder. */
    for(index_t j = 0; j < (1L << k); j++)
        b[j] = sub[0*K+j];
    b[(1L << k)] = F::one; // monic
    array_quorem_monic(0, n, f, (1L << k)+1, b, q, t); // note +1 here
    for(index_t i = n; i < (1L << k); i++)
        t[i] = F::zero;
       
    /* Now work down the levels. */
    F *r = t;
    F *w = s;
    for(index_t l = 1; l <= k; l++) {

        /* Invariant: r contains 2^{l-1} remainders of degree 2^{k-l+1}. */
        /* Compute next-level remainders to w. */

        array_scatter(1L << l, 1L << (k-l), (1L << (k-l))+1, // note +1 here
                      sub + l*K, b);

        for(index_t j = 0; j < (1L << l); j++)
            b[((1L << (k-l))+1)*j + (1L << (k-l))] = F::one; // monic, note +1

        array_interleave(1L << (l-1), 1L << (k-l+1), r, r, a);

        array_quorem_monic(l, 1L << (k-l+1), a, (1L << (k-l))+1, b, q, w);

        /* Transpose r and w. */
        F *temp = w;
        w = r;
        r = temp;
    }
    assert(r != w);

    /* Copy result. */
    for(index_t i = 0; i < p; i++)
        f_u[i] = r[i];

    array_delete(t);
    array_delete(s);
    array_delete(q);
    array_delete(b);
    array_delete(a);
    
    array_delete(sub);
}

/* Build a Lagrange polynomial for p points via subproducts. */

template <typename F>
void array_lagrange_sub(index_t p, const F *u, const F *c, F *f)
{
    assert(p >= 1);
    index_t k = ceil_log2(p); // Evaluate at 2^k points
    index_t K = 1L << k;
    
    /* Ascend a subproduct tree with 2^{k+1}-1 nodes. */

    /* Level l = 0,1,...,k has 2^l nodes. */

    /* The full polynomials are monic of degree 2^{k-l},
     * the Lagrange polynomials are of degree 2^{k-l}-1. */

    F *t0 = array_allocate<F>(1L << (k+1));
    F *t1 = array_allocate<F>(1L << (k+1));
    F *t2 = array_allocate<F>(1L << (k+1));
    F *t3 = array_allocate<F>(1L << (k+1));
    F *x = array_allocate<F>(1L << (k+1));
    F *y = array_allocate<F>(1L << (k+1));
    F *z = array_allocate<F>(1L << (k+1));

    F *f0 = t0;
    F *f1 = t1;
    F *l0 = t2;
    F *l1 = t3;

    /* Prepare the leaf polynomials. */
    for(index_t i = 0; i < K; i++) {
        f0[i] = i < p ? -u[i] : F::zero;
           // the monic polynomial x-u[i]
        l0[i] = i < p ? c[i] : F::zero; 
           // coefficient for the Lagrange term that omits x-u[i]
    }

    /* Process the internal nodes one level at a time from bottom to top. */
    for(index_t l = k-1; l >= 0; l--) {

        // Multiply full nodes at level l+1 to get full node at level l. 

        array_gather2 (l, k-l, k-(l+1), f0, z);
        array_scatter2(l, k-(l+1), k-l, z, x);
        array_gather2 (l, k-l, k-(l+1), f0, z, k-(l+1));
        array_scatter2(l, k-(l+1), k-l, z, y);

        array_monic2(l, k-l, k-(l+1), x);
        array_monic2(l, k-l, k-(l+1), y);

        ss_mul(l, k-l, x, y, f1);  

        array_add_one2(l, k-l, f1); // cancel monic -1 wrap by adding 1

        // Mix the full and Lagrange nodes at level l+1
        // to get Lagrange node at level l. 

        array_gather2 (l, k-l, k-(l+1), f0, z);
        array_scatter2(l, k-(l+1), k-l, z, x);
        array_gather2 (l, k-l, k-(l+1), l0, z, k-(l+1));
        array_scatter2(l, k-(l+1), k-l, z, y);

        array_monic2(l, k-l, k-(l+1), x);

        ss_mul(l, k-l, x, y, l1);

        array_gather2 (l, k-l, k-(l+1), l0, z);
        array_scatter2(l, k-(l+1), k-l, z, x);
        array_gather2 (l, k-l, k-(l+1), f0, z, k-(l+1));
        array_scatter2(l, k-(l+1), k-l, z, y);

        array_monic2(l, k-l, k-(l+1), y);

        ss_mul(l, k-l, x, y, z);

        array_add(1L << k, z, 1L << k, l1, 1L << k, l1);

        F *temp = f0;
        f0 = f1;
        f1 = temp;
        temp = l0;
        l0 = l1;
        l1 = temp;
    }
    for(index_t i = 0; i < p; i++)
        f[i] = l0[i+K-p];

    array_delete(z);
    array_delete(y);
    array_delete(x);
    array_delete(t3);
    array_delete(t2);
    array_delete(t1);
    array_delete(t0);
}

/* Interpolation with the subproduct algorithm. */

template <typename F>
void array_poly_interpolate(index_t p, const F *u, const F *v, F *f)
{
    F *c = array_allocate<F>(p);
    for(index_t i = 0; i < p; i++)
        c[i] = F::one;
    array_lagrange_sub(p, u, c, f);
    array_poly_batch_eval(p, f, p, u, c);
    for(index_t i = 0; i < p; i++)
        c[i] = v[i]*c[i].inv();
    array_lagrange_sub(p, u, c, f);   
    array_delete(c);
}


/**************************************************** Class for polynomials. */

template <typename F>
class Poly
{
    bool               init;
    index_t            cap;  
    std::shared_ptr<F> ptr;
    
    void reset_to_cap(index_t c) {
        assert(c >= 0);
        cap = c;
        ptr.reset(array_allocate<F>(c), array_deleter<F>());
    }
    void trim(void) {
        assert(init);
        index_t d = degree();
        if(d < 0)
            d = 0;
        if(cap != d+1) {
            F *a = array_allocate<F>(d+1);
            array_copy(d+1, ptr.get(), a);
            cap = d+1;
            ptr.reset(a, array_deleter<F>());
        }
    }   
    Poly(index_t c) {
        init = true;
        reset_to_cap(c);
    }

public:
    Poly() : init(false) { }
    Poly(const Poly& other) {
        init = other.init;
        if(init) {
            cap = other.cap;
            ptr = other.ptr;
        }
    }
    Poly(const F& other) {
        init = true;
        reset_to_cap(1);
        ptr.get()[0] = other;
    }
    ~Poly() { }   
    
    bool initialized(void) const { return init; }
    index_t capacity(void) const {
        assert(init);
        return cap;
    }
    index_t degree(void) const {
        assert(init);
        return array_poly_deg(cap, ptr.get());
    }
    F& operator[](index_t i) {
        assert(init && i >= 0 && i < cap);
        return ptr.get()[i];
    }
    F& operator[](index_t i) const {
        assert(init && i >= 0 && i < cap);
        return ptr.get()[i];
    }
    F eval(const F u) const {
        assert(init);
        return array_poly_eval(cap, ptr.get(), u);
    }
    F operator()(const F u) const { return eval(u); }
    
                    
    Poly& operator=(const Poly rhs) {
        assert(rhs.init);
        init = true;
        cap  = rhs.cap;
        ptr  = rhs.ptr;
        return *this;
    }
    friend Poly operator+(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        Poly c(a.cap > b.cap ?
               a.cap : b.cap);
        array_add(a.cap, a.ptr.get(),
                  b.cap, b.ptr.get(),
                  c.cap, c.ptr.get());
        return c;
    }
    friend Poly operator-(const Poly& a) {
        assert(a.init);
        Poly b(a.cap);
        array_neg(a.cap, a.ptr.get(),
                  b.cap, b.ptr.get());
        return b;
    }
    friend Poly operator-(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        Poly c(a.cap > b.cap ? a.cap : b.cap);
        array_sub(a.cap, a.ptr.get(),
                  b.cap, b.ptr.get(),
                  c.cap, c.ptr.get());
        return c;
    }
    friend Poly operator*(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        index_t ad = a.degree();
        index_t bd = b.degree();
        index_t deg = ad < bd ? bd : ad;
        deg = deg <= 0 ? 1 : deg;
        index_t d = ceil_log2(deg+1)+1;
        index_t D = 1L << d;                
        F *aa = array_allocate<F>(D);
        F *bb = array_allocate<F>(D);
        F *cc = array_allocate<F>(D);
        array_scatter(1, a.cap, D, a.ptr.get(), aa);
        array_scatter(1, b.cap, D, b.ptr.get(), bb);
        ss_mul(0, d, aa, bb, cc);
        deg = array_poly_deg(D, cc);
        if(deg < 0) { deg = 0; }
        Poly c(deg + 1);
        array_copy(deg + 1, cc, c.ptr.get());
        array_delete(cc);
        array_delete(bb);
        array_delete(aa);
        return c;
    }
    friend void quorem(const Poly& a, const Poly& b, Poly& q, Poly& r) {
        assert(a.init && b.init);
        index_t ad = a.degree();
        index_t bd = b.degree();
        assert(bd >= 0); // guard against divide by zero
        index_t qd = ad - bd;
        if(qd < 0) {
            q = Poly(F::zero);
            r = b;          
        } else {
            q = Poly(qd+1);
            r = Poly(bd+1); // must be able to save a zero remainder
            array_poly_quorem(ad + 1, a.ptr.get(), 
                              bd + 1, b.ptr.get(), 
                              q.ptr.get(), 
                              r.ptr.get());
        }
    }
    friend Poly operator/(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        Poly q, r;
        quorem(a, b, q, r);
        return q;
    }
    friend Poly operator%(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        Poly q, r;
        quorem(a, b, q, r);
        return r;
    }
    bool divides(const Poly &p) { return p % *this == Poly(F::zero); }
    
    friend Poly operator*(const Poly& a, const F s) {
        assert(a.init);
        Poly b(a.cap);
        array_scalar_mul(a.cap, a.ptr.get(),
                         s,
                         b.cap, b.ptr.get());
        return b;
    }
    friend Poly operator*(const F s, const Poly& a) {
        return a*s;
    }
    friend bool operator==(const Poly& a, const Poly& b) {
        assert(a.init && b.init);
        return array_poly_eq(a.cap, a.ptr.get(),
                             b.cap, b.ptr.get());
    }
    friend bool operator!=(const Poly& a, const Poly& b) {
        return !(a==b);
    }

    static Poly x(index_t deg) {
        assert(deg >= 0);
        Poly<F> a(deg + 1);
        array_zero(a.cap, a.ptr.get());
        a.ptr.get()[deg] = F::one;
        return a;
    }
    static Poly x(void) { return x(1); }

    static Poly interpolate(index_t n, const F *u, const F*v) {
        assert(n >= 1);
        Poly<F> p(n);
        array_poly_interpolate(n, u, v, p.ptr.get());
        return p;
    }

    void batch_eval(index_t p, const F *u, F *f_u) {
        assert(p >= 1);
        array_poly_batch_eval(cap, ptr.get(), p, u, f_u);
    }

    Poly take(index_t k) const {
        // Take k+1 largest-degree coefficients of f
        // and return them as a degree k polynomial;
        // return the zero polynomial if k is negative
        // (with degree capped at degree of polynomial)

        index_t d = array_poly_deg(cap, ptr.get());
        if(k >= 0) {
            if(k > d)
                k = d;
            Poly r(k+1);
            array_copy(k+1, ptr.get() + d - k, r.ptr.get());
            return r;
        } else {
            return Poly(F::zero);
        }
    }

    static Poly rand(index_t n) {
        assert(n >= 1);
        Poly a(n);
        array_rand(a.cap, a.ptr.get());
        return a;
    }
};

template <typename F>
std::ostream& operator<<(std::ostream& out, const Poly<F>& p)
{
    index_t deg = p.degree();
    if(deg < 0) {
        out << "0";
    } else {
        for(index_t i = deg; i >= 0; i--) {
            if(p[i] != F::zero) {
                if(i < deg)
                    out << " + ";
                if(p[i] != F::one || i == 0)
                    out << p[i];
                if(i > 0)
                    out << "x^{" << i << "}";
            }
        }
    }
    return out;
}

/********************************************* Extended Euclidean algorithm. */

template <typename F>
void gcd_recursive(index_t lvl,
                   const Poly<F> r_0,
                   const Poly<F> r_1,
                   index_t k,
                   index_t &h,
                   Poly<F> &s_h,
                   Poly<F> &t_h,
                   Poly<F> &s_hp1,
                   Poly<F> &t_hp1)
{
    // note: base case goes down to very low degree
    // should resort to a simpler algorithm earlier
    if(r_1.degree() < 0 || k < r_0.degree() - r_1.degree()) {
        h = 0;
        s_h = Poly<F>(F::one);
        t_h = Poly<F>(F::zero);
        s_hp1 = Poly<F>(F::zero);
        t_hp1 = Poly<F>(F::one);
        return;
    } else {
        if(k == 0 && k == r_0.degree() - r_1.degree()) {
            h = 1;
            s_h = Poly<F>(F::zero);
            t_h = Poly<F>(F::one);
            s_hp1 = Poly<F>(F::one);
            t_hp1 = Poly<F>(-r_0[r_0.degree()]*(r_1[r_1.degree()].inv()));
            return;
        }
    }
    index_t d = (k+1)/2;
    assert(d >= 1);

    Poly<F> f_s_h, f_t_h, f_s_hp1, f_t_hp1;
    index_t f_h;

    Poly<F> f_r_0 = r_0.take(2*d-2);
    Poly<F> f_r_1 = r_1.take(2*d-2-(r_0.degree()-r_1.degree()));
    
    gcd_recursive(lvl+1,
                  f_r_0,
                  f_r_1,
                  d-1,
                  f_h,
                  f_s_h,
                  f_t_h,
                  f_s_hp1,
                  f_t_hp1);

    index_t j = f_h + 1;
    index_t delta = f_t_hp1.degree();

    Poly<F> m_r_0 = r_0.take(2*k);
    Poly<F> m_r_1 = r_1.take(2*k-(r_0.degree()-r_1.degree()));

    // 4 x mul        
    Poly<F> r_jm1 = m_r_0*f_s_h   + m_r_1*f_t_h;
    Poly<F> r_j   = m_r_0*f_s_hp1 + m_r_1*f_t_hp1;

    if(r_j.degree() < 0 || 
       k < delta + r_jm1.degree() - r_j.degree()) {
        h = j-1;      
        s_h = f_s_h;
        t_h = f_t_h;
        s_hp1 = f_s_hp1;
        t_hp1 = f_t_hp1;
        return;
    }

    // quorem
    Poly<F> q_j, r_jp1;
    quorem(r_jm1, r_j, q_j, r_jp1); 
    
    index_t dstar = k - delta - (r_jm1.degree() - r_j.degree());
    assert(dstar >= 0);

    Poly<F> l_r_0 = r_j.take(2*dstar);
    Poly<F> l_r_1 = r_jp1.take(2*dstar-(r_j.degree()-r_jp1.degree()));
    Poly<F> l_s_h, l_t_h, l_s_hp1, l_t_hp1;
    index_t l_h;
    gcd_recursive(lvl+1,
                  l_r_0,
                  l_r_1,
                  dstar,
                  l_h,
                  l_s_h,
                  l_t_h,
                  l_s_hp1,
                  l_t_hp1);

    h = l_h + j;

    // 10 x mul 
    // (could use Strassen's here to get rid of at least one)
    Poly<F> u11 = f_s_hp1;
    Poly<F> u12 = f_t_hp1;
    Poly<F> u21 = f_s_h-q_j*f_s_hp1;
    Poly<F> u22 = f_t_h-q_j*f_t_hp1;
    s_h   = l_s_h*u11 + l_t_h*u21;
    t_h   = l_s_h*u12 + l_t_h*u22;
    s_hp1 = l_s_hp1*u11 + l_t_hp1*u21;
    t_hp1 = l_s_hp1*u12 + l_t_hp1*u22;   
}

template <typename F>
void gcd(const Poly<F> &r_0,
         const Poly<F> &r_1,
         index_t k,
         index_t &h,
         Poly<F> &r_h,
         Poly<F> &s_h,
         Poly<F> &t_h,
         Poly<F> &r_hp1,
         Poly<F> &s_hp1,
         Poly<F> &t_hp1)
{
    gcd_recursive(0, r_0, r_1, k, h, s_h, t_h, s_hp1, t_hp1);
    r_h   = r_0*s_h   + r_1*t_h;
    r_hp1 = r_0*s_hp1 + r_1*t_hp1;  

    assert(r_h   == r_0*s_h   + r_1*t_h);
    assert(r_hp1 == r_0*s_hp1 + r_1*t_hp1);
    if(r_hp1.degree() < 0) {
        assert(r_h.divides(r_0));
        assert(r_h.divides(r_1));
    } else {
        assert(r_0.degree() - r_h.degree() <= k);
        assert(r_0.degree() - r_hp1.degree() > k);
    }
}

/****************************************** Reed--Solomon encoding/decoding. */

template <typename F>
void rs_encode(index_t d, index_t e, F *p, F *q)
{
    assert(d >= 0 && d+1 <= e);
    Poly<F> f = Poly<F>::x(d);
    for(index_t i = 0; i <= d; i++)
        f[i] = p[i];
    F *u = array_allocate<F>(e);
    for(index_t i = 0; i < e; i++)
        u[i] = F(i);
    f.batch_eval(e, u, q);
    array_delete(u);
}

template <typename F>
void corrupt(index_t d, index_t e, F *f)
{
    assert(d >= 0 && d+1 <= e);
    index_t n = (e-d-1)/2;
    index_t *q = array_allocate<index_t>(e);
    randperm(e, q);
    for(index_t i = 0; i < n; i++)
        f[q[i]] = F::rand();
    array_delete(q);
}

template <typename F>
bool rs_decode(index_t d, index_t e, F *r, F *p)
{
    // Gao's decoder
    assert(d >= 0 && d+1 <= e);

    F *c = array_allocate<F>(e+1);
    F *u = array_allocate<F>(e+1);
    F *f = array_allocate<F>(e+1);
    Poly<F> g0, g1, r0, s0, t0, r1, s1, t1;

    for(index_t i = 0; i < e+1; i++) {      
        u[i] = F(i);
        c[i] = (i == e) ? F::one : F::zero;
    }
    array_lagrange_sub(e+1, u, c, f);
    g0 = Poly<F>::x(e);
    for(index_t i = 0; i < e; i++)
        g0[i] = f[i];
    g1 = Poly<F>::interpolate(e, u, r);

    index_t k = (e-d-1)/2;
    index_t h;

    gcd(g0, g1, k, h, r0, s0, t0, r1, s1, t1);

    bool success = r1.degree() < (e+d+1)/2;
    if(success) {
        success = (t1.degree() <= r1.degree()) && 
                  (r1.degree() - t1.degree() <= d);
        if(success) {
            quorem(r1, t1, r0, s0);
            success = s0.degree() < 0;
            if(success) {
                array_zero(d+1, p);
                index_t dd = r0.degree();
                for(index_t i = 0; i <= dd; i++)
                    p[i] = r0[i];
            }
        }
    }
    array_delete(f);
    array_delete(u);
    array_delete(c);    

    return success;
}

template <typename F>
bool rs_decode_xy(index_t d, index_t e, F *x, F *y, Poly<F> &p)
{
    // Gao's decoder
    assert(d >= 0 && d+1 <= e);

    F *u = array_allocate<F>(e+1);
    F *c = array_allocate<F>(e+1);
    F *f = array_allocate<F>(e+1);
    Poly<F> g0, g1, r0, s0, t0, r1, s1, t1;

    for(index_t i = 0; i < e+1; i++) {
        u[i] = (i < e) ? x[i] : F::zero;
        c[i] = (i == e) ? F::one : F::zero;
    }
    array_lagrange_sub(e+1, u, c, f);
    g0 = Poly<F>::x(e);
    for(index_t i = 0; i < e; i++)
        g0[i] = f[i];

    array_delete(f);
    array_delete(u);
    array_delete(c);

    g1 = Poly<F>::interpolate(e, x, y);

    index_t k = (e-d-1)/2;  
    index_t h;

    gcd(g0, g1, k, h, r0, s0, t0, r1, s1, t1);

    bool success = r1.degree() < (e+d+1)/2;
    if(success) {
        success = (t1.degree() <= r1.degree()) && 
                  (r1.degree() - t1.degree() <= d);
        if(success) {
            quorem(r1, t1, r0, s0);
            success = s0.degree() < 0;
            if(success)
                p = r0;
        }
    }

    return success;
}


/****************************************************************** Testing. */

/* Test polynomial GCD. */

template <typename F>
void test_gcd(void)
{
    for(index_t n = 1; n < 15; n++) {
        for(index_t m = 1; m < n; m++) {
            for(index_t g = 0; g <= m; g++) {               
                for(index_t k = 0; k <= n; k++) {               
                    for(index_t rr = 0; rr < 3; rr++) {
                        Poly<F> r0, r1, z0;
                        Poly<F> x = Poly<F>::rand(g+1);
                        if(x[g] == F::zero)
                            x[g] = F::one;
                        
                        if(n > g) {
                            Poly<F> z0 = Poly<F>::rand(n-g+1);
                            if(z0[n-g] == F::zero)
                                z0[n-g] = F::one;
                            r0 = x*z0;
                        } else {
                            r0 = x;
                        }
                        if(m > g) {
                            Poly<F> z1 = Poly<F>::rand(m-g+1);
                            if(z1[m-g] == F::zero)
                                z1[m-g] = F::one;
                            r1 = x*z1;
                        } else {
                            r1 = x;
                        }
                        
                        assert(n == r0.degree());
                        assert(m == r1.degree());
                        
                        Poly<F> rh, sh, th, rhp1, shp1, thp1;
                        index_t h;
                        metric_push(m_default);
                        gcd(r0, r1, k, h, rh, sh, th, rhp1, shp1, thp1);
                        metric_pop(m_default,
                                   "gcd: n = %3ld, m = %3ld, g = %3ld, "
                                   "k = %3ld, rr = %3ld", 
                                    n, m, g, k, rr);
                    }
                }
            }
        }
    }
}

/* Performance-test gcd computation. */

template <typename F>
void test_gcd_perf(void)
{
    for(index_t u = 0; u <= 20; u++) {
        index_t g = 1L << u;
        index_t n = 2*g;
        index_t m = 2*g;
        index_t k = n;
        for(index_t rr = 0; rr < 1; rr++) {
            Poly<F> r0, r1, z0;
            Poly<F> x = Poly<F>::rand(g+1);
            if(x[g] == F::zero)
                x[g] = F::one;
            
            if(n > g) {
                Poly<F> z0 = Poly<F>::rand(n-g+1);
                if(z0[n-g] == F::zero)
                    z0[n-g] = F::one;
                r0 = x*z0;
            } else {
                r0 = x;
            }
            if(m > g) {
                Poly<F> z1 = Poly<F>::rand(m-g+1);
                if(z1[m-g] == F::zero)
                    z1[m-g] = F::one;
                r1 = x*z1;
            } else {
                r1 = x;
            }
            
            assert(n == r0.degree());
            assert(m == r1.degree());
            
            Poly<F> rh, sh, th, rhp1, shp1, thp1;
            index_t h;
            metric_push(m_default);
            gcd(r0, r1, k, h, rh, sh, th, rhp1,shp1, thp1);
            metric_pop(m_default,
                       "gcd: n = %7ld, m = %7ld, g = %7ld, "
                       "k = %7ld, rr = %3ld",
                       n, m, g, k, rr);
        }
    }
}

/* Test Reed--Solomon encoding/decoding. */

template <typename F>
void test_rs(void) 
{
    index_t maxl = 26;
    index_t trials = 5;
    for(index_t d = 0; d <= maxl; d++) {
        for(index_t e = d+1; e <= maxl; e++) {
            for(index_t rr = 0; rr < trials; rr++) {
                F *src = array_allocate<F>(d+1);
                F *dec = array_allocate<F>(d+1);
                F *enc = array_allocate<F>(e);

                metric_push(m_default);
                array_rand(d+1, src);
                rs_encode(d, e, src, enc);
                corrupt(d, e, enc);
                assert(rs_decode(d, e, enc, dec));
                assert(array_eq(d+1, src, dec));
                metric_pop(m_default,
                           "rs: d = %8ld, e = %8ld, rr = %5ld", 
                            d, e, rr);
                array_delete(enc);
                array_delete(dec);
                array_delete(src);
            }
        }
    }
}

/* Performance-test Reed--Solomon encoding/decoding. */

template <typename F>
void test_rs_perf(void) 
{
    for(index_t dd = 0; dd <= 20; dd++) {
        index_t d = 1L << dd;
        index_t e = 2*d;
        F *src = array_allocate<F>(d+1);
        F *dec = array_allocate<F>(d+1);
        F *enc = array_allocate<F>(e);
        
        array_rand(d+1, src);

        metric_push(m_default);
        rs_encode(d, e, src, enc);
        metric_pop(m_default, "encode:  d = %10ld, e = %10ld", d, e);

        metric_push(m_default);
        corrupt(d, e, enc);
        metric_pop(m_default, "corrupt: d = %10ld, e = %10ld", d, e);

        metric_push(m_default);
        assert(rs_decode(d, e, enc, dec));
        metric_pop(m_default, "decode:  d = %10ld, e = %10ld", d, e);

        metric_push(m_default);
        assert(array_eq(d+1, src, dec));
        metric_pop(m_default, "eqtest:  d = %10ld, e = %10ld", d, e);
        
        array_delete(enc);
        array_delete(dec);
        array_delete(src);
    }
}

/* Performance-test polynomial multiplication. */

template <typename F>
void test_mul_perf(void)
{
    for(index_t k = 0; k <= 25; k++) {
        index_t n = 1L << k;
        Poly<F> f = Poly<F>::rand(n);
        Poly<F> g = Poly<F>::rand(n);
        metric_push(m_default);
        Poly<F> h = f*g;
        metric_pop(m_default,
                   "mul: k = %2ld, n = %11ld", k, n);
    }
}

/* Test polynomial quotient and remainder. */

template <typename F>
void test_quorem(void) 
{   
    index_t trials = 1;
    for(index_t u = 6; u <= 25; u++) {
        index_t n = (1L << u);
        for(index_t t = 0; t < trials; t++) {
            Poly<F> a = Poly<F>::rand(n);
            Poly<F> b = Poly<F>::rand(n);
            Poly<F> c = Poly<F>::rand(b.degree());
            Poly<F> d = a*b + c;
            Poly<F> q, r;
            metric_push(m_default);
            quorem(d, b, q, r);
            metric_pop(m_default,
                       "quorem: n = %10ld, t = %2ld", 
                        n, t);
            assert(q == a);
            assert(r == c);
        }
    }
}

/* Test polynomial evaluation and interpolation. */

template <typename F>
void test_eval_interp(void) 
{

    index_t trials = 1;
    for(index_t u = 6; u <= 20; u++) {
        index_t n = (1L << u);
        for(index_t r = 0; r < trials; r++) {
            F *u = array_allocate<F>(n);
            F *v = array_allocate<F>(n);
            F *w = array_allocate<F>(n);
            
            for(index_t i = 0; i < n; i++) {
                u[i] = F(i);
                v[i] = F::rand();
            }

            metric_push(m_default);
            Poly<F> p = Poly<F>::interpolate(n, u, v);
            metric_pop(m_default,
                       "interpolate: n = %12ld, r = %2ld", n, r);

            metric_push(m_default);
            p.batch_eval(n, u, w);
            metric_pop(m_default,
                       "evaluate:    n = %12ld, r = %2ld", n, r);

            assert(array_eq(n, v, w));
            array_delete(w);
            array_delete(v);
            array_delete(u);
        }
    }
}

/* Performance-test for Montgomery multiplication. */

template <typename F>
__global__ 
void ker_montgomery_test(F *d_a, F *d_b)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;

    F a = d_a[v];
    F b = d_b[v];
    for(index_t i = 0; i < (1L << 16); i++)
        a = a * b;
    d_b[v] = a;
}

template <typename F>
void gpu_montgomery_test(void)
{
    index_t db = 1024;
    index_t dg = 4096;
    index_t repeats = 1L << 16;

    F *a = array_allocate<F>(db*dg);
    F *b = array_allocate<F>(db*dg);
    F *r = array_allocate<F>(db*dg);
    F *d_a = dev_array_allocate<F>(db*dg);
    F *d_b = dev_array_allocate<F>(db*dg);

    array_rand(db*dg, a);
    array_rand(db*dg, b);  

    dev_array_upload(db*dg, a, d_a);
    dev_array_upload(db*dg, b, d_b);

    metric_push(m_default);

    ker_montgomery_test<<<dg,db>>>(d_a, d_b);
    CUDA_SYNC;

    double mul_count = repeats*db*dg;
    double time = metric_time(m_default);
    metric_pop(m_default,
               "montgomery_test: %8.2lfGmul/s",
               mul_count/1e9/(time/1000.0));

    dev_array_download(db*dg, d_b, r);

    for(index_t i = 0; i < db; i++) {
        F x = a[i];
        F y = b[i];
        for(index_t j = 0; j < repeats; j++)
            x = x*y;
        assert(r[i] == x);
    }

    dev_array_delete(d_a);
    dev_array_delete(d_b);

    array_delete(r);
    array_delete(b);
    array_delete(a);    
}

/******************************************************************** Tests. */
 
template <typename F>
void test_F()
{
    gpu_montgomery_test<F>();
    test_mul_perf<F>();
    test_quorem<F>();
    test_eval_interp<F>();
    test_gcd<F>();
    test_rs<F>();
    test_gcd_perf<F>();
    test_rs_perf<F>();
}

/*****************************************************************************/
/********************************************************** GPU subroutines. */
/*****************************************************************************/

/***************************************************** Transpose on the GPU. */

/* Transpose dimensions u and v of data, u < v. 
 * Note: parameters refer to dimensions of the OUTPUT. */

template <typename F>
__global__
void ker_transpose(index_t N,   // volume of data array
                   index_t m_v, // length of dimension v
                   index_t M_v, // product of lengths of dimensions below v
                   index_t m_u, // length of dimension u
                   index_t M_u, // product of lengths of dimensions below u
                   F *d_in,
                   F *d_out) 
{
    index_t b = blockDim.x*blockIdx.x+threadIdx.x;
    index_t b_v_hi = b/M_v;
    index_t b_hi   = b_v_hi/m_v;        // above dimension v
    index_t b_v    = b_v_hi%m_v;        // dimension v
    index_t b_mid  = (b%M_v)/(m_u*M_u); // between dimensions u and v (exclusive)
    index_t b_u    = (b/M_u)%m_u;       // dimension u
    index_t b_lo   = b%M_u;             // below dimension u

    // build the index with u and v transposed
    index_t a = b_hi*m_v*M_v + b_u*((M_v*m_v)/m_u) + b_mid*m_v*M_u + b_v*M_u + b_lo;

    // copy data
    if(b < N)
        d_out[b] = d_in[a];
}

template <typename F>
void transpose(index_t N,   // volume of data array
               index_t m_v, // length of dimension v
               index_t M_v, // product of lengths of dimensions below v
               index_t m_u, // length of dimension u
               index_t M_u, // product of lengths of dimensions below u
               F *d_data,
               F *d_scratch,
               bool leave_to_scratch = false)
{

    metric_push(m_gpu_transpose);

    // Sanity-check the parameters
    assert(N >= 0 && M_u >= 0 && M_v >= 0 && m_u >= 0 && m_v >= 0);
    assert(N % (m_v*M_v) == 0);
    assert(N % (m_u*M_u) == 0);
    assert(M_v % (m_u*M_u) == 0);

    index_t db = 1024;
    index_t dg = (N+db-1)/db;
    
    ker_transpose<<<dg,db>>>(N, m_v, M_v, m_u, M_u, d_data, d_scratch);
    CUDA_SYNC;
    if(!leave_to_scratch)
        dev_array_copy(N, d_scratch, d_data);    

    double time = metric_time(m_gpu_transpose);
    double trans_bytes = 2*N*sizeof(F);
    metric_pop(m_gpu_transpose,
               "transpose: "
               "N = %12ld (%6.2lfGiB/s)",
               N,
               trans_bytes/(1L << 30)/(time/1000.0));
}

/************************************************** Stretch from N to N x P. */

template <typename F>
__global__
void ker_stretch(index_t N,    // volume of input array
                 index_t P,    // stretch factor (output has dimensions N x P)
                 const F *d_in,
                 F *d_out)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    index_t v_lo = v / P;
    if(v < N*P)
        d_out[v] = d_in[v_lo];
}

template <typename F>
void stretch(index_t N,     // volume of input array
             index_t P,     // stretch factor (output has dimensions N x P)
             const F *d_in,
             F *d_out)
{
    // Sanity-check the parameters
    assert(N >= 0 && P >= 1 && d_in != d_out);

    index_t db = 1024;
    index_t dg = (N*P+db-1)/db;
    ker_stretch<<<dg,db>>>(N, P, d_in, d_out);
    CUDA_SYNC;
}

/****************************************** Multiplicative scan for scalars. */

/* 
 * Based on 
 *
 * Sengupta, Harris, Garland, Owens,
 * "Efficient parallel scan algorithms for many-core GPUs"
 * http://www.idav.ucdavis.edu/publications/print_pub?pub_id=1041
 *
 * and
 *
 * Luitjens, "Faster parallel reductions on Kepler"
 * https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 *
 */

/* 
 * Scan inside a warp with warp shuffle (intrinsics).
 *
 */

template <typename F>
__device__ F scan_warp(F warp)
{
    const uint lane = threadIdx.x & 31;

    for(int delta = 1; delta < 32; delta*=2) {
        F acc = warp*F(__shfl_up_sync(0xFFFFFFFFU, warp.raw(), delta), true);
        warp = (lane < delta) ? warp : acc;
    }

    F acc = F(__shfl_up_sync(0xFFFFFFFFU, warp.raw(), 1), true);
    warp = (lane == 0) ? F(1) : acc;

    return warp;
}

/* 
 * Scans in blocks of 1024 threads, each block processes a segment of
 * 2*1024 = 2048 consecutive scalar4-vectors. Uses shared memory to transfer
 * scan results between warps. Intra-warp scanning via the subroutine above. 
 *
 */

template <typename F>
__global__ void ker_scan_segments(scalar4_t *d_in)
{
    extern __shared__ scalar_t s[];

    const uint lane = threadIdx.x & 31;    // 0,1,...,31
    const uint warpid = threadIdx.x >> 5;  // 0,1,...,31
    
    d_in += 2*blockDim.x*blockIdx.x;      // + 2*32*32 * blockIdx.x
    
    scalar4_t warp4_lo = d_in[threadIdx.x];
    scalar4_t warp4_hi = d_in[threadIdx.x + 32*32];

    F warp_lo = F(warp4_lo.x, true)*
                F(warp4_lo.y, true)*
                F(warp4_lo.z, true)*
                F(warp4_lo.w, true);
    F warp_lo_ex = scan_warp<F>(warp_lo);
    warp_lo = warp_lo*warp_lo_ex;

    if(lane == 31)
        s[warpid] = warp_lo.raw();
    __syncthreads();

    if(warpid == 0)
        s[threadIdx.x] = scan_warp<F>(F(s[threadIdx.x],true)).raw();
    __syncthreads();
    
    warp4_lo.x = (F(warp4_lo.x, true)*warp_lo_ex*F(s[warpid],true)).raw();
    warp4_lo.y = (F(warp4_lo.y, true)*F(warp4_lo.x, true)).raw();
    warp4_lo.z = (F(warp4_lo.z, true)*F(warp4_lo.y, true)).raw();
    warp4_lo.w = (F(warp4_lo.w, true)*F(warp4_lo.z, true)).raw();
    __syncthreads();

    if(threadIdx.x == 32*32-1)
        s[0] = F(warp4_lo.w, true).raw();
    __syncthreads();

    F low_total = F(s[0], true);
    __syncthreads();

    F warp_hi = F(warp4_hi.x, true)*
                F(warp4_hi.y, true)*
                F(warp4_hi.z, true)*
                F(warp4_hi.w, true);    
    F warp_hi_ex = scan_warp<F>(warp_hi);
    warp_hi = warp_hi*warp_hi_ex;

    if(lane == 31)
        s[warpid] = warp_hi.raw();
    __syncthreads();

    if(warpid == 0)
        s[threadIdx.x] = scan_warp<F>(F(s[threadIdx.x],true)).raw();
    __syncthreads();

    warp4_hi.x = (F(warp4_hi.x, true)*warp_hi_ex*F(s[warpid],true)*low_total).raw();
    warp4_hi.y = (F(warp4_hi.y, true)*F(warp4_hi.x, true)).raw();
    warp4_hi.z = (F(warp4_hi.z, true)*F(warp4_hi.y, true)).raw();
    warp4_hi.w = (F(warp4_hi.w, true)*F(warp4_hi.z, true)).raw();

    d_in[threadIdx.x] = warp4_lo;
    d_in[threadIdx.x + 32*32] = warp4_hi;
}

/* 
 * Gather segment totals from last element of each segment.
 *
 */

template <typename F>
__global__
void ker_segment_totals(index_t nseg, scalar4_t *d_in, F *d_rec)
{
    index_t seg = blockDim.x*blockIdx.x + threadIdx.x;
    d_rec[seg] = seg < nseg ? F(d_in[2048*seg+2047].w, true) : F(0);
}

/* 
 * Add (scanned) segment totals to each segment.
 *
 */

template <typename F>
__global__ void ker_add_totals(scalar4_t *d_in, F *d_rec)
{
    index_t seg = blockIdx.x;
    F total = (seg == 0) ? F(1) : d_rec[seg-1];

    d_in += 2*blockDim.x*blockIdx.x;      // + 2*32*32 * blockIdx.x
    scalar4_t lo = d_in[threadIdx.x];
    scalar4_t hi = d_in[threadIdx.x + 32*32];
    lo.x = (F(lo.x, true)*total).raw();
    lo.y = (F(lo.y, true)*total).raw();
    lo.z = (F(lo.z, true)*total).raw();
    lo.w = (F(lo.w, true)*total).raw();
    hi.x = (F(hi.x, true)*total).raw();
    hi.y = (F(hi.y, true)*total).raw();
    hi.z = (F(hi.z, true)*total).raw();
    hi.w = (F(hi.w, true)*total).raw();
    d_in[threadIdx.x] = lo;
    d_in[threadIdx.x + 32*32] = hi;
}

/*
 * Top-level (recursive) scan procedure. 
 *
 */

template <typename F>
void dev_scalar_scan(index_t n, F *d_data, index_t level = 0)
{
    assert(n % 8192 == 0);
    if(n == 0)
        return;
    index_t dg = n/8192;
    index_t db = 32*32;
    index_t sm = sizeof(int)*32;

    if(level == 0)
        metric_push(m_gpu_scan);

    ker_scan_segments<F><<<dg,db,sm>>>((scalar4_t *) d_data);
    CUDA_SYNC;

    if(n > 8192) {
        index_t n_seg = n/8192;
        index_t n_rec = ((n_seg + 8192-1)/8192)*8192;
        F *d_rec = dev_array_allocate<F>(n_rec);
        assert(n_rec >= n_seg && n_rec % 32 == 0);

        ker_segment_totals<F><<<n_rec/32,32>>>(n_seg, 
                                               (scalar4_t *) 
                                               d_data, d_rec);
        CUDA_SYNC;
    

        dev_scalar_scan(n_rec, d_rec, level+1);

        ker_add_totals<F><<<n_seg,32*32>>>((scalar4_t *) d_data, d_rec);
        CUDA_SYNC;

        dev_array_delete(d_rec);
    }

    if(level == 0) {
        double trans_bytes = sizeof(F)*(n > 8192 ? 4 : 2)*n;
        double time = metric_time(m_gpu_scan);
        metric_pop(m_gpu_scan,
                   "dev_scalar_scan: "
                   "n = %ld, dg = %ld, db = %ld, sm = %ld (%6.2lfGiB/s)", 
                    n, dg, db, sm,
                    trans_bytes/((double)(1L << 30))/(time/1000.0));
    }
}

template <typename F>
__global__
void ker_diff(index_t R, index_t R_round, F *d_x0, F *d_diff)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    F i = v % R_round;
    index_t j = v / R_round;
    F x0 = d_x0[j];
    F out = F(1);
    if(i < R)
        out = x0 - F(i);
    d_diff[v] = out;
}

/***************************************** Pointwise modular multiplication. */

template <typename F>
__global__ 
void ker_mod_mul(index_t N, F *d_a, F *d_b, F *d_c)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    if(v < N) {
        F a = d_a[v];
        F b = d_b[v];
        F c = a*b;
        d_c[v] = c;
    }
}

template <typename F>
__global__ 
void ker_mod_mul4(index_t N, scalar4_t *d_a4, scalar4_t *d_b4, scalar4_t *d_c4)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    if(v < N) {
        scalar4_t a4 = d_a4[v];
        scalar4_t b4 = d_b4[v];
        scalar4_t c4;
        c4.x = (F(a4.x, true)*F(b4.x, true)).raw();
        c4.y = (F(a4.y, true)*F(b4.y, true)).raw();
        c4.z = (F(a4.z, true)*F(b4.z, true)).raw();
        c4.w = (F(a4.w, true)*F(b4.w, true)).raw();
        d_c4[v] = c4;
    }
}

template <typename F>
void gpu_mod_mul(index_t N, F *d_a, F *d_b, F *d_c)
{
    if(N % 4 == 0) {
        index_t db = 1024;
        index_t dg = (N/4+db-1)/db;
        ker_mod_mul4<F><<<dg,db>>>(N/4, 
                                   (scalar4_t *) d_a, 
                                   (scalar4_t *) d_b, 
                                   (scalar4_t *) d_c);
        CUDA_SYNC;
    } else {
        index_t db = 1024;
        index_t dg = (N+db-1)/db;
        ker_mod_mul<<<dg,db>>>(N, d_a, d_b, d_c);
        CUDA_SYNC;
    }  
}


/**************************************************** Modular sum reduction. */

template <typename F>
__global__ 
void ker_mod_add_reduce(index_t p, index_t N, F *d_in, F *d_out)
{
    index_t v = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    if(v < p) {
        F s = F(0);
        for(index_t j = 0; j < N; j++)
            s = s + d_in[v*N+j];
        d_out[v] = s;
    }
}

template <typename F>
void gpu_mod_add_reduce(index_t p, index_t N, F *d_in, F *d_out)
{
    index_t db = 1024;
    index_t dg = (p+db-1)/db;
    ker_mod_add_reduce<<<dg,db>>>(p, N, d_in, d_out);
    CUDA_SYNC;
}


/*****************************************************************************/
/*********** Yates's algorithm & Strassen's matrix multiplication algorithm. */
/*****************************************************************************/

/* Strassen's decomposition for <2,2,2>. */

// NOTE *CONVENTIONAL* INDEX SYMMETRY: ij' jk' i'k
// indices agree with rows in order 00,01,10,11

#define STRASSEN_A     1, 0, 1, 0, 1,-1, 0,  \
                       0, 0, 0, 0, 1, 0, 1,  \
                       0, 1, 0, 0, 0, 1, 0,  \
                       1, 1, 0, 1, 0, 0,-1

#define STRASSEN_B     1, 1, 0,-1, 0, 1, 0,  \
                       0, 0, 1, 0, 0, 1, 0,  \
                       0, 0, 0, 1, 0, 0, 1,  \
                       1, 0,-1, 0, 1, 0, 1

#define STRASSEN_B_T   1, 1, 0,-1, 0, 1, 0,  \
                       0, 0, 0, 1, 0, 0, 1,  \
                       0, 0, 1, 0, 0, 1, 0,  \
                       1, 0,-1, 0, 1, 0, 1

#define STRASSEN_C     1, 0, 0, 1,-1, 0, 1,  \
                       0, 0, 1, 0, 1, 0, 0,  \
                       0, 1, 0, 1, 0, 0, 0,  \
                       1,-1, 1, 0, 0, 1, 0

#define STRASSEN_A_TEMPLATE   < F, STRASSEN_A >
#define STRASSEN_B_TEMPLATE   < F, STRASSEN_B >
#define STRASSEN_B_T_TEMPLATE < F, STRASSEN_B_T >
#define STRASSEN_C_TEMPLATE   < F, STRASSEN_C >

int strassenA[] = { STRASSEN_A };
int strassenB[] = { STRASSEN_B };
int strassenC[] = { STRASSEN_C };

#define TEMPLATE_F4X7 \
    <typename F, \
     int A00, int A01, int A02, int A03, int A04, int A05, int A06, \
     int A10, int A11, int A12, int A13, int A14, int A15, int A16, \
     int A20, int A21, int A22, int A23, int A24, int A25, int A26, \
     int A30, int A31, int A32, int A33, int A34, int A35, int A36>

#define TEMPLATE_F4X7_INST \
    <F, \
     A00, A01, A02, A03, A04, A05, A06, \
     A10, A11, A12, A13, A14, A15, A16, \
     A20, A21, A22, A23, A24, A25, A26, \
     A30, A31, A32, A33, A34, A35, A36>

/* CPU implementations. */
                          
/* Yates's algorithm. */

template <typename F>
void yates_level(index_t k, 
                 index_t t, index_t s, const int *A, 
                 index_t l, const F *in, F *out)
{
    /* 
     * Level l = 0,1,...,k-1.
     * Input has size s^{k-l}t^{l}, output has size s^{k-l-1}t^{l+1}.
     * The base matrix A has size t by s.
     *
     */

    index_t base = index_pow(t,l);    
    index_t in_size = index_pow(s,k-l)*base;
    index_t out_size = in_size*t/s;
    for(index_t v = 0; v < out_size; v++) {
        index_t v_lsd = v%base;     // digits up to l-1 
        index_t v_l = (v/base)%t;   // digit l
        index_t v_msd = (v/base)/t; // digits starting from l+1
        index_t u_base = v_lsd + v_msd*base*s;      
        F y = F(0);
        for(index_t j = 0; j < s; j++) {
            int a = A[v_l*s+j];
            if(a != 0) {
                F z = in[u_base + base*j];
                if(a == 1)
                    y = y + z;
                else
                    y = y - z;
            }
        }
        out[v] = y;
    }
}

template <typename F>
void yates(index_t k, index_t t, index_t s, 
           const int *a, const F *in, F *out)
{
    assert(t >= 1 && s >= 1 && k >= 0);
    index_t m = t > s ? t : s;
    index_t buf_size = index_pow(m, k);
    index_t in_size = index_pow(s, k);
    index_t out_size = index_pow(t, k);
    F *b0 = array_allocate<F>(buf_size);
    F *b1 = array_allocate<F>(buf_size);
    array_copy(in_size, in, b0);
    for(index_t l = 0; l < k; l++) {
        yates_level(k, t, s, a, l, b0, b1);
        F *bt = b0;
        b0 = b1;
        b1 = bt;
    }
    array_copy(out_size, b0, out);
    array_delete(b1);
    array_delete(b0);
}

/* Polynomial extension of sparse Yates's algorithm. */

template <typename F>
void yates_poly(index_t k, index_t l,
                index_t t, index_t s, int *A,
                index_t m, const index_t *in_pos, const F *in_val, 
                F z0, F *out)
{

    /* The base matrix A has size t by s. */

    assert(t >= 1 && s >= 1 && k >= 0 && t >= s);
    assert(l >= 0 && l <= k);

    index_t lagrange_size = index_pow(t, k-l);
    index_t coeff_size = index_pow(s, k-l);
    index_t base_size = index_pow(s, l);

    int *A_T = (int *) array_allocate<int>(s*t);
    F *lagrange = array_allocate<F>(lagrange_size);
    F *f = array_allocate<F>(lagrange_size);
    F *coeff = array_allocate<F>(coeff_size);
    F *base = array_allocate<F>(base_size);

    for(index_t i = 0; i < t; i++) 
        for(index_t j = 0; j < s; j++) 
            A_T[j*t+i] = A[i*s+j];

    if(z0.value() < lagrange_size) {
        for(index_t i = 0; i < lagrange_size; i++)
            lagrange[i] = (i == z0.value()) ? F(1) : F(0);
    } else {
        f[0] = F(1);
        for(index_t i = 1; i < lagrange_size; i++)
            f[i] = f[i-1]*F(i);
        F p = F(1);
        for(index_t i = 0; i < lagrange_size; i++)
            p = p*(z0-F(i));
        for(index_t i = 0; i < lagrange_size; i++) {
            F q = F(1);
            q = (((lagrange_size-(i+1))%2) == 0) ? q : -q;
            q = q*(z0-F(i));
            q = q*f[i];
            q = q*f[lagrange_size-(i+1)];
            lagrange[i] = p*(q.inv());
        }
    }
    yates(k-l, s, t, A_T, lagrange, coeff);

    for(index_t j = 0; j < base_size; j++) 
        base[j] = 0;
    for(index_t u = 0; u < m; u++) {
        index_t j = in_pos[u];
        index_t j_lsd = j % base_size;
        index_t j_msd = j / base_size;
        assert(j_msd >= 0 && j_msd < coeff_size);       
        base[j_lsd] = base[j_lsd] + coeff[j_msd]*in_val[u];
    }
    yates(l, t, s, A, base, out);

    array_delete(base);
    array_delete(coeff);
    array_delete(f);
    array_delete(lagrange);
    array_delete(A_T);
}

/* GPU implementations. */

/* 4-by-7 and 7-by-4 Yates transforms. */

template TEMPLATE_F4X7
__global__
void ker_47_yates_level4(index_t grid_size, index_t lo_mod, 
                         scalar4_t *d_in, scalar4_t *d_out)
{
    index_t z = blockDim.x*blockIdx.x+threadIdx.x;
    if(z < grid_size) {
        index_t z_lo = z % lo_mod; 
        index_t z_hi = z / lo_mod;
        index_t u_base = z_hi*7*lo_mod + z_lo;
        index_t v_base = z_hi*4*lo_mod + z_lo;
            
        // Read in 7 values (4-vectorized)
        scalar4_t r0 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r1 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r2 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r3 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r4 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r5 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r6 = d_in[u_base];

        // Compute 4 values (4-vectorized)

        scalar4_t s0; 
        scalar4_t s1;
        scalar4_t s2;
        scalar4_t s3;

        YATES_47_BATCH;
        
        // Write 4 values (4-vectorized)

        d_out[v_base] = s0; v_base += lo_mod;
        d_out[v_base] = s1; v_base += lo_mod;
        d_out[v_base] = s2; v_base += lo_mod;
        d_out[v_base] = s3;
    }   
}

template TEMPLATE_F4X7
void gpu_47_yates4(index_t D, index_t k, index_t p, F *d_in, F *d_out, F *d_s0, F *d_s1)
{   
    assert(k >= 1);
    assert(p % 4 == 0);

    F *d_b0;
    F *d_b1;    
    for(index_t l = 0; l < k; l++) {
        metric_push(m_gpu_yates);

        // input  has dimensions D x 4^{l} x 7 x 7^{k-l-1} x P/4 x 4
        // output has dimensions D x 4^{l} x 4 x 7^{k-l-1} x P/4 x 4

        // grid dimensions       D x 4^{l} x   x 7^{k-l-1} x P/4
        // each thread will read 7 and write 4 scalar4s
        
//      index_t b0_size   = D*index_pow(4,l)*7*index_pow(7,k-l-1)*p;
        index_t b1_size   = D*index_pow(4,l)*4*index_pow(7,k-l-1)*p;
        index_t grid_size = D*index_pow(4,l)*1*index_pow(7,k-l-1)*p/4;
        index_t lo_mod    =                    index_pow(7,k-l-1)*p/4;

        if(l == 0) {
            d_b0 = d_in;
            d_b1 = d_s0;
        }
        if(l == 1)
            d_b1 = d_s1;
        if(l == k-1)
            d_b1 = d_out;

        index_t db = 1024;
        index_t dg = (grid_size + db - 1)/db;
               
        ker_47_yates_level4 TEMPLATE_F4X7_INST
            <<<dg,db>>>(grid_size,
                        lo_mod,
                        (scalar4_t *) d_b0, 
                        (scalar4_t *) d_b1);
        CUDA_SYNC;

        F *d_t = d_b0;
        d_b0 = d_b1;
        d_b1 = d_t;

        double time = metric_time(m_gpu_yates);
        double trans_bytes = grid_size*(7+4)*sizeof(scalar4_t);
        metric_pop(m_gpu_yates,
                   "ker_47_yates4: "
                   "D = %2ld, k = %2ld, l = %2ld, p = %5ld (%6.2lfGiB/s)",
                   D,
                   k,
                   l,
                   p,
                   trans_bytes/(1L << 30)/(time/1000.0));
    }
}

template TEMPLATE_F4X7
__global__
void ker_74_yates_level4(index_t grid_size, index_t lo_mod, 
                         scalar4_t *d_in, scalar4_t *d_out)
{
    index_t z = blockDim.x*blockIdx.x+threadIdx.x;
    if(z < grid_size) {
        index_t z_lo = z % lo_mod; 
        index_t z_hi = z / lo_mod;
        index_t u_base = z_hi*4*lo_mod + z_lo;
        index_t v_base = z_hi*7*lo_mod + z_lo;
            
        // Read in 4 values (4-vectorized)
        scalar4_t r0 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r1 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r2 = d_in[u_base]; u_base += lo_mod;
        scalar4_t r3 = d_in[u_base]; 

        // Compute 7 values (4-vectorized)

        scalar4_t s0; 
        scalar4_t s1;
        scalar4_t s2;
        scalar4_t s3;
        scalar4_t s4;
        scalar4_t s5;
        scalar4_t s6;

        YATES_74_BATCH;

        // Write 7 values (4-vectorized)

        d_out[v_base] = s0; v_base += lo_mod;
        d_out[v_base] = s1; v_base += lo_mod;
        d_out[v_base] = s2; v_base += lo_mod;
        d_out[v_base] = s3; v_base += lo_mod;
        d_out[v_base] = s4; v_base += lo_mod;
        d_out[v_base] = s5; v_base += lo_mod;
        d_out[v_base] = s6;
    }
}

template TEMPLATE_F4X7
void gpu_74_yates4(index_t D, index_t k, index_t p, F *d_in, F *d_out, F *d_s)
{   
    assert(k >= 1);
    assert(p % 4 == 0);

    F *d_b0;
    F *d_b1;    
    for(index_t l = 0; l < k; l++) {
        metric_push(m_gpu_yates);

        // input  has dimensions D x 4^{k-l-1} x 4 x 7^{l} x P/4 x 4
        // output has dimensions D x 4^{k-l-1} x 7 x 7^{l} x P/4 x 4

        // grid dimensions       D x 4^{k-l-1} x   x 7^{l} x P/4
        // each thread will read 4 and write 7 scalar4s
        
//      index_t b0_size   = D*index_pow(4,k-l-1)*4*index_pow(7,l)*p;
        index_t b1_size   = D*index_pow(4,k-l-1)*7*index_pow(7,l)*p;
        index_t grid_size = D*index_pow(4,k-l-1)*1*index_pow(7,l)*p/4;
        index_t lo_mod    =                        index_pow(7,l)*p/4;

        if(l == 0) {
            d_b0 = d_in;
            d_b1 = (k % 2 == 0) ? d_s : d_out;
        }
        if(l == 1)
            d_b1 = (k % 2 == 0) ? d_out : d_s;

        index_t db = 1024;
        index_t dg = (grid_size + db - 1)/db;

        ker_74_yates_level4 TEMPLATE_F4X7_INST
            <<<dg,db>>>(grid_size,
                        lo_mod,
                        (scalar4_t *) d_b0, 
                        (scalar4_t *) d_b1);
        CUDA_SYNC;

        F *d_t = d_b0;
        d_b0 = d_b1;
        d_b1 = d_t;

        double time = metric_time(m_gpu_yates);
        double trans_bytes = grid_size*(7+4)*sizeof(scalar4_t);
        metric_pop(m_gpu_yates,
                   "ker_74_yates4: "
                   "D = %2ld, k = %2ld, l = %2ld, p = %5ld (%6.2lfGiB/s)",
                   D,
                   k,
                   l,
                   p,
                   trans_bytes/(1L << 30)/(time/1000.0));
    }
}

/* Inner 4x4 multiply for matrix multiplication. */

template <typename F>
__global__
void ker_4x4_mul(index_t grid_size, 
                 scalar4_t *d_a, scalar4_t *d_b, scalar4_t *d_c)
{
    index_t z = blockDim.x*blockIdx.x+threadIdx.x;
    if(z < grid_size) {
        scalar4_t a0 = d_a[z + 0*grid_size];
        scalar4_t a1 = d_a[z + 1*grid_size];
        scalar4_t a2 = d_a[z + 2*grid_size];
        scalar4_t a3 = d_a[z + 3*grid_size];
        scalar4_t b0 = d_b[z + 0*grid_size];
        scalar4_t b1 = d_b[z + 1*grid_size];
        scalar4_t b2 = d_b[z + 2*grid_size];
        scalar4_t b3 = d_b[z + 3*grid_size];
        scalar4_t c0;
        scalar4_t c1;
        scalar4_t c2;
        scalar4_t c3;
        MUL_4X4_BATCH;
    }   
}

template <typename F>
__global__
void ker_4x4_mulT(index_t grid_size, 
                  scalar4_t *d_a, scalar4_t *d_b, scalar4_t *d_c)
{
    index_t z = blockDim.x*blockIdx.x+threadIdx.x;
    if(z < grid_size) {
        scalar4_t a0 = d_a[z + 0*grid_size];
        scalar4_t a1 = d_a[z + 1*grid_size];
        scalar4_t a2 = d_a[z + 2*grid_size];
        scalar4_t a3 = d_a[z + 3*grid_size];
        scalar4_t b0 = d_b[z + 0*grid_size];
        scalar4_t b1 = d_b[z + 1*grid_size];
        scalar4_t b2 = d_b[z + 2*grid_size];
        scalar4_t b3 = d_b[z + 3*grid_size];
        scalar4_t c0;
        scalar4_t c1;
        scalar4_t c2;
        scalar4_t c3;
        MUL_4X4_RIGHT_TRANSPOSE_BATCH;
    } 
}

template <typename F>
void gpu_4x4_mul(index_t transpose_right, index_t N,
                 F *d_a, F *d_b, F *d_c)
{
    metric_push(m_gpu_mul);

    index_t grid_size = N/(4*4);
    index_t db = 1024;
    index_t dg = (grid_size + db - 1)/db;

    if(transpose_right) {
        ker_4x4_mulT<F><<<dg,db>>>(grid_size,
                                   (scalar4_t *) d_a,
                                   (scalar4_t *) d_b, 
                                   (scalar4_t *) d_c);
        CUDA_SYNC;
    } else {
        ker_4x4_mul<F><<<dg,db>>>(grid_size,
                                  (scalar4_t *) d_a,
                                  (scalar4_t *) d_b, 
                                  (scalar4_t *) d_c);
        CUDA_SYNC;
    }

    double time = metric_time(m_gpu_mul);
    double trans_bytes = grid_size*(3*4*4)*sizeof(F);
    double mul_count = grid_size*64;
    metric_pop(m_gpu_mul,
               "gpu_4x4_mul: "
               "N = %12ld (%6.2lfGiB/s, %6.2lfGmul/s)",
               N,
               trans_bytes/(1L << 30)/(time/1000.0),
               mul_count/(1e9)/(time/1000.0));
}

/* Strassen's algorithm with inner plain 4x4 matrix multiply. */

template <typename F>
void gpu_strassen(index_t transpose_right, index_t k, index_t p, 
                  F *d_a, F *d_b, F *d_c, 
                  F *d_aS, F *d_bS, F *d_cS)
{

    metric_push(m_gpu_strassen);
    push_time();

    // Input has dimensions 4^{k} x P

    assert(k >= 2);
    int n = 1L << k;

    // Transpose operands
    transpose(n*n*p, p, 4, 4, 1, d_a, d_aS);
    if(d_a != d_b)
        transpose(n*n*p, p, 4, 4, 1, d_b, d_bS);
    // Operands have dimensions 4^{k-1} x P x 4

    // Strassen-transform operands
    gpu_74_yates4 STRASSEN_A_TEMPLATE (4, k-2, p*4, d_a, d_aS, d_cS);
    if(transpose_right)
        gpu_74_yates4 STRASSEN_B_T_TEMPLATE (4, k-2, p*4, d_b, d_bS, d_cS);
    else
        gpu_74_yates4 STRASSEN_B_TEMPLATE   (4, k-2, p*4, d_b, d_bS, d_cS);

    // Mid-layer data has dimensions 4 x 7^{k-2} x P x 4
    // Execute 4 x 4 matrix mul on MSD x LSD
    gpu_4x4_mul(transpose_right, 4*index_pow(7,k-2)*p*4, d_aS, d_bS, d_cS);

    // Inverse-Strassen-transform the result
    gpu_47_yates4 STRASSEN_C_TEMPLATE (4, k-2, p*4, d_cS, d_bS, 
                                       (k % 2 == 0) ? d_aS : d_bS, 
                                       (k % 2 == 0) ? d_bS : d_aS);

    // Transpose result
    transpose(n*n*p, 4, p, p, 1, d_bS, d_c, true);
    // Result has dimensions 4^{k} x P

    // Restore original input (unless overwritten by result)
    if(d_a != d_c)
        transpose(n*n*p, 4, p, p, 1, d_a, d_aS);
    if(d_b != d_a && d_b != d_c)
        transpose(n*n*p, 4, p, p, 1, d_b, d_bS);   
    // Input has dimensions 4^{k} x P 

    gpu_mm_time += pop_time();

    double time = metric_time(m_gpu_strassen);
    double mul_sim_count = index_pow(7,k)*p;
    double mul_actual_count = index_pow(7,k-2)*p*64;
    double mul_cubic_count = index_pow(8,k-2)*p*64;
    metric_pop(m_gpu_strassen,
               "strassen: k = %2ld, p = %5ld: "
               "%6.2lfGmul/s (Strassen), "
               "%6.2lfGmul/s (actual), "
               "%6.2lfGmul/s (cubic)",
               k,
               p,
               mul_sim_count/(1e9)/(time/1000.0),
               mul_actual_count/(1e9)/(time/1000.0),
               mul_cubic_count/(1e9)/(time/1000.0));
}

/*****************************************************************************/
/********************************************* The \binom{6}{2}-linear form. */
/*****************************************************************************/

/* CPU implementations. */

/* The brute-force implementation. */

template <typename F>
F binom62_linear(index_t k, F *chi)
{
    index_t n = 1L << k;

    F *chi_ab = chi +  0*n*n;
    F *chi_ac = chi +  1*n*n;
    F *chi_ad = chi +  2*n*n;
    F *chi_ae = chi +  3*n*n;
    F *chi_af = chi +  4*n*n;
    F *chi_bc = chi +  5*n*n;
    F *chi_bd = chi +  6*n*n;
    F *chi_be = chi +  7*n*n;
    F *chi_bf = chi +  8*n*n;
    F *chi_cd = chi +  9*n*n;
    F *chi_ce = chi + 10*n*n;
    F *chi_cf = chi + 11*n*n;
    F *chi_de = chi + 12*n*n;
    F *chi_df = chi + 13*n*n;
    F *chi_ef = chi + 14*n*n;

    F r = F(0);
    for(index_t a = 0; a < n; a++) {
        for(index_t b = 0; b < n; b++) {
            for(index_t c = 0; c < n; c++) {
                for(index_t d = 0; d < n; d++) {
                    for(index_t e = 0; e < n; e++) {
                        for(index_t f = 0; f < n; f++) {
                            F p = F(1);
                            p = p * chi_ab[a*n+b];
                            p = p * chi_ac[a*n+c];
                            p = p * chi_ad[a*n+d];
                            p = p * chi_ae[a*n+e];
                            p = p * chi_af[a*n+f];
                            p = p * chi_bc[b*n+c];
                            p = p * chi_bd[b*n+d];
                            p = p * chi_be[b*n+e];
                            p = p * chi_bf[b*n+f];
                            p = p * chi_cd[c*n+d];
                            p = p * chi_ce[c*n+e];
                            p = p * chi_cf[c*n+f];
                            p = p * chi_de[d*n+e];
                            p = p * chi_df[d*n+f];
                            p = p * chi_ef[e*n+f];
                            r = r + p;
                        }
                    }
                }
            }
        }
    }
    return r;
}

/* Evaluation of the Camelot polynomial. */

/* Prepare the n-by-n coefficient arrays alpha, beta, gamma. */

template <typename F>
void coeff(index_t k, 
           F x0,
           F *alpha, F *beta, F *gamma)
{
    index_t n = 1L << k;
    index_t R = index_pow(7, k);
    
    F *alphai = array_allocate<F>(n*n);
    F *betai  = array_allocate<F>(n*n);
    F *gammai = array_allocate<F>(n*n);
    F *lagrange = array_allocate<F>(R);
    F *f = array_allocate<F>(R);

    if(x0.value() < R) {
        for(index_t i = 0; i < R; i++)
            lagrange[i] = (i == x0.value()) ? F(1) : F(0);
    } else {
        f[0] = F(1);
        for(index_t i = 1; i < R; i++)
            f[i] = f[i-1]*F(i);
        F p = F(1);
        for(index_t i = 0; i < R; i++)
            p = p * (x0 - F(i));
        for(index_t i = 0; i < R; i++) {
            F q = F(1);
            q = (((R-(i+1))%2) == 0) ? q : -q;
            q = q * (x0 - F(i));
            q = q * f[i];
            q = q * f[R-(i+1)];
            lagrange[i] = p * (q.inv());
        }
    }
    yates(k, 4, 7, strassenA, lagrange, alphai);
    yates(k, 4, 7, strassenB, lagrange, betai);
    yates(k, 4, 7, strassenC, lagrange, gammai);

    for(index_t i = 0; i < n*n; i++) {
        index_t ii = from_interleaved(k, i);
        alpha[ii] = alphai[i];
        beta[ii]  = betai[i];
        gamma[ii] = gammai[i];
    }

    array_delete(f);
    array_delete(lagrange);

    array_delete(gammai);
    array_delete(betai);
    array_delete(alphai);
}

/* Single-point evaluation. */

template <typename F>
F binom62_linear_poly(index_t k, F *chi, F x0)
{
    index_t n = 1L << k;

    // Get working memory
    F *alpha = array_allocate<F>(n*n);  
    F *beta = array_allocate<F>(n*n);   
    F *gamma = array_allocate<F>(n*n);  
    F *A = array_allocate<F>(n*n);  
    F *B = array_allocate<F>(n*n);  
    F *C = array_allocate<F>(n*n);  
    F *H = array_allocate<F>(n*n);  
    F *K = array_allocate<F>(n*n);  
    F *L = array_allocate<F>(n*n);  
    F *Q = array_allocate<F>(n*n);

    F *chi_ab = chi +  0*n*n;
    F *chi_ac = chi +  1*n*n;
    F *chi_ad = chi +  2*n*n;
    F *chi_ae = chi +  3*n*n;
    F *chi_af = chi +  4*n*n;
    F *chi_bc = chi +  5*n*n;
    F *chi_bd = chi +  6*n*n;
    F *chi_be = chi +  7*n*n;
    F *chi_bf = chi +  8*n*n;
    F *chi_cd = chi +  9*n*n;
    F *chi_ce = chi + 10*n*n;
    F *chi_cf = chi + 11*n*n;
    F *chi_de = chi + 12*n*n;
    F *chi_df = chi + 13*n*n;
    F *chi_ef = chi + 14*n*n;

    // Compute the coefficient arrays
    coeff(k, x0, alpha, beta, gamma);

    // Compute the matrices H, K, L, A, B, C, Q 
    // Caveat: we do not use Strassen's here
    for(index_t a = 0; a < n; a++) {
        for(index_t d = 0; d < n; d++) {
            F s = F(0);
            for(index_t ep = 0; ep < n; ep++)
                s = s + alpha[d*n+ep]*chi_ae[a*n+ep]*chi_de[d*n+ep];
            H[a*n+d] = s;           
        }
    }
    for(index_t b = 0; b < n; b++) {
        for(index_t e = 0; e < n; e++) {
            F s = F(0);
            for(index_t fp = 0; fp < n; fp++)
                s = s + beta[e*n+fp]*chi_bf[b*n+fp]*chi_ef[e*n+fp];
            K[b*n+e] = s;
        }
    }
    for(index_t c = 0; c < n; c++) {
        for(index_t f = 0; f < n; f++) {
            F s = F(0);
            for(index_t dp = 0; dp < n; dp++)
                s = s + gamma[dp*n+f]*chi_cd[c*n+dp]*chi_df[dp*n+f];
            L[c*n+f] = s;
        }
    }
    for(index_t a = 0; a < n; a++) {
        for(index_t b = 0; b < n; b++) {
            F s = F(0);
            for(index_t d = 0; d < n; d++)
                s = s + chi_ad[a*n+d]*chi_bd[b*n+d]*H[a*n+d];
            A[a*n+b] = s;
        }
    }
    for(index_t b = 0; b < n; b++) {
        for(index_t c = 0; c < n; c++) {
            F s = F(0);
            for(index_t e = 0; e < n; e++)
                s = s + chi_be[b*n+e]*chi_ce[c*n+e]*K[b*n+e];
            B[b*n+c] = s;
        }
    }
    for(index_t a = 0; a < n; a++) {
        for(index_t c = 0; c < n; c++) {
            F s = F(0);
            for(index_t f = 0; f < n; f++)
                s = s +chi_af[a*n+f]*chi_cf[c*n+f]*L[c*n+f];
            C[a*n+c] = s;
        }
    }
    for(index_t a = 0; a < n; a++) {
        for(index_t b = 0; b < n; b++) {
            F s = F(0);
            for(index_t c = 0; c < n; c++)
                s = s + chi_ac[a*n+c]*chi_bc[b*n+c]*B[b*n+c]*C[a*n+c];
            Q[a*n+b] = s;
        }
    }

    // Compute the evaluation at x0
    F Px0 = F(0);
    for(index_t a = 0; a < n; a++)
        for(index_t b = 0; b < n; b++)
            Px0 = Px0 + chi_ab[a*n+b]*A[a*n+b]*Q[a*n+b];

    // Release working memory
    array_delete(Q);
    array_delete(L);
    array_delete(K);
    array_delete(H);
    array_delete(C);
    array_delete(B);
    array_delete(A);
    array_delete(gamma);
    array_delete(beta);
    array_delete(alpha);

    return Px0;
}

/* GPU implementation. */

/* Evaluation of the Camelot polynomial. */

/* Subroutines for preparing look-up arrays for Lagrange coefficients. */

template <typename F>
__global__ void ker_iota(F *d_data)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    F z = (v == 0) ? F(1) : F(v); 
    d_data[v] = z;
}

template <typename F>
__global__ void ker_mod_inv(F *d_in, F *d_out)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    d_out[v] = d_in[v].inv();
}

template <typename F>
__global__ void ker_mod_conv(index_t R, F *d_in, F *d_out)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    if(v < R)
        d_out[v] = d_in[v] * d_in[R-(v+1)];
}

template <typename F>
__global__ void ker_lagrange_base(index_t p, index_t R, index_t first, 
                                  F *d_f, F *d_lagrange_base)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    if(v < p && first + v >= R && first + v < 4*R)
        d_lagrange_base[v] = d_f[first + v] * (d_f[first + v - R].inv()); 
}

template <typename F>
struct coeff_tab
{
    F *d_factorial_table;
    F *d_finvconv_table;
    F *d_inverse_table;
};

template <typename F>
void gpu_coeff_precompute(index_t k,
                          coeff_tab<F> &pre)
{
    index_t R = index_pow(7, k);
    index_t R_round = ((R+8192-1)/8192)*8192;
    
    pre.d_factorial_table = dev_array_allocate<F>(4*R_round);
    pre.d_inverse_table   = dev_array_allocate<F>(4*R_round);
    pre.d_finvconv_table  = dev_array_allocate<F>(R_round);

    // Prepare look-up tables for inverses and factorials
    {
    index_t db = 1024;
    assert(3*R_round % db == 0);
    index_t dg = (4*R_round)/db;
    ker_iota<<<dg,db>>>(pre.d_factorial_table);
    CUDA_SYNC;
    ker_mod_inv<<<dg,db>>>(pre.d_factorial_table, pre.d_inverse_table);
    CUDA_SYNC;
    dev_scalar_scan<F>(4*R_round,
                       pre.d_factorial_table);
    }

    // Prepare convolved inverse factorial table
    {
    F *d_finv = dev_array_allocate<F>(R_round);
    index_t db = 1024;
    assert(R_round % db == 0);
    index_t dg = R_round/db;
    ker_mod_inv<<<dg,db>>>(pre.d_factorial_table, d_finv);
    CUDA_SYNC;
    ker_mod_conv<<<dg,db>>>(R, d_finv, pre.d_finvconv_table);
    CUDA_SYNC;
    dev_array_delete(d_finv);
    }
}

template <typename F>
void gpu_coeff_release(coeff_tab<F> &pre)
{
    dev_array_delete(pre.d_finvconv_table);
    dev_array_delete(pre.d_inverse_table);
    dev_array_delete(pre.d_factorial_table);
}

/* Build Lagrange coefficients using precomputed arrays. */

template <typename F>
__global__
void ker_lagrange(index_t R, 
                  index_t P,
                  index_t first,
                  F *d_finvconv, 
                  F *d_inv,
                  F *d_Ld, 
                  F *d_L)
{
    index_t v = blockDim.x*blockIdx.x+threadIdx.x;
    if(v < R*P) {
        index_t i = v / P;  // i = 0,1,...,R-1 indexes the Lagrange divisor
        index_t j = v % P;  // j = 0,1,...,P-1 indexes the point (=first+j)
        F q;
        if(first + j < R ||    // Lagrange polynomials are 0/1-valued
           first + j >= 4*R) { // Don't-care values used to pad P 
            q = (i == first+j) ? F(1) : F(0);
        } else {
            // Here we need to compute the value of the polynomial
            q = d_Ld[j];
            q = q * d_finvconv[i];
            q = q * d_inv[first+j-i];
            if(((R-(i+1))&1) == 1)
                q = -q;
        }
        d_L[v] = q;
    }
}

/* Prepare the n-by-n arrays alpha, beta, gamma for p points. */

template <typename F>
void gpu_coeff(index_t p, index_t k, index_t first,
               coeff_tab<F> &pre,
               F *d_alpha, 
               F *d_beta, 
               F *d_gamma,
               F *d_lagrange_base,
               F *d_lagrange,
               F *d_s)
{
    index_t R = index_pow(7, k);
    assert(first >= 0);

    {
    index_t db = 32*32;
    index_t dg = (p+db-1)/db;
    ker_lagrange_base<<<dg,db>>>(p, 
                                 R, 
                                 first, 
                                 pre.d_factorial_table, 
                                 d_lagrange_base);
    CUDA_SYNC;
    }            

    {       
    index_t db = 32*32;
    index_t dg = (p*R + db-1)/db;
    ker_lagrange<<<dg,db>>>(R, p, first, 
                            pre.d_finvconv_table, pre.d_inverse_table, 
                            d_lagrange_base, d_lagrange);
    CUDA_SYNC;
    }

    // d_lagrange has dimensions 7^k x P

    gpu_47_yates4 STRASSEN_A_TEMPLATE (1, k, p, d_lagrange, d_alpha, d_s, d_s + p*4*(R/7));
    gpu_47_yates4 STRASSEN_B_TEMPLATE (1, k, p, d_lagrange, d_beta,  d_s, d_s + p*4*(R/7));
    gpu_47_yates4 STRASSEN_C_TEMPLATE (1, k, p, d_lagrange, d_gamma, d_s, d_s + p*4*(R/7));

    // Coefficient arrays have dimensions 4^k x P 
    
}

/* Evaluation at p points. */

template <typename F>
void gpu_binom62_linear_poly(index_t p_in, index_t k, 
                             coeff_tab<F> &pre,
                             F *d_chi_base, 
                             index_t first, 
                             F *Px0_in,
                             F *d_scratch)
{
    index_t n = 1L << k;

    metric_push(m_gpu_eval);
    
    // Round the number of evaluation points to a power of 2.
    index_t p = max(4L, 1L << ceil_log2(p_in));
    F *Px0 = array_allocate<F>(p);

    assert(p >= 4); // will work with 4-vectorization over p

    F *d_chi_ab = d_chi_base +  0*n*n;
    F *d_chi_ac = d_chi_base +  1*n*n;
    F *d_chi_ad = d_chi_base +  2*n*n;
    F *d_chi_ae = d_chi_base +  3*n*n;
    F *d_chi_af = d_chi_base +  4*n*n;
    F *d_chi_bc = d_chi_base +  5*n*n;
    F *d_chi_bd = d_chi_base +  6*n*n;
    F *d_chi_be = d_chi_base +  7*n*n;
    F *d_chi_bf = d_chi_base +  8*n*n;
    F *d_chi_cd = d_chi_base +  9*n*n;
    F *d_chi_ce = d_chi_base + 10*n*n;
    F *d_chi_cf = d_chi_base + 11*n*n;
    F *d_chi_de = d_chi_base + 12*n*n;
    F *d_chi_df = d_chi_base + 13*n*n;
    F *d_chi_ef = d_chi_base + 14*n*n;
    
    F *d_chi   = d_scratch + 0*n*n*p;
    F *d_alpha = d_scratch + 1*n*n*p;
    F *d_beta  = d_scratch + 2*n*n*p;
    F *d_gamma = d_scratch + 3*n*n*p; 

    metric_push(m_gpu_eval_detail);

    index_t R = index_pow(7, k);
    index_t R_round = ((R+8192-1)/8192)*8192;
    index_t p_round = ((p+8192-1)/8192)*8192;
    
    F *d_lagrange_base = d_scratch + 4*n*n*p;
    F *d_lagrange      = d_scratch + 4*n*n*p + p_round;
    F *d_s             = d_scratch + 4*n*n*p + p_round + p*R_round;

    // Compute the coefficient arrays
    gpu_coeff(p, k, first, pre, 
              d_alpha, d_beta, d_gamma,
              d_lagrange_base, d_lagrange, d_s);

    metric_pop(m_gpu_eval_detail, "coeff");

    // Coefficient arrays now have dimensions 4^k x P

    // Get some scratch
    F *d_s1 = d_scratch + 4*n*n*p;
    F *d_s2 = d_scratch + 4*n*n*p +   4*index_pow(7,k-2)*p*4;   
    F *d_s3 = d_scratch + 4*n*n*p + 2*4*index_pow(7,k-2)*p*4;   

    // Compute the matrices H, K, L
    F *d_H = d_alpha;
    F *d_K = d_beta;  
    F *d_L = d_gamma;  

    metric_push(m_gpu_eval_detail);

    stretch(index_pow(4,k), p, d_chi_de, d_chi);
    gpu_mod_mul(n*n*p, d_alpha, d_chi, d_H);
    stretch(index_pow(4,k), p, d_chi_ae, d_chi);
    gpu_strassen(1, k, p, d_chi, d_H, d_H, d_s1, d_s2, d_s3);

    stretch(index_pow(4,k), p, d_chi_ef, d_chi);
    gpu_mod_mul(n*n*p, d_beta, d_chi, d_K);
    stretch(index_pow(4,k), p, d_chi_bf, d_chi);
    gpu_strassen(1, k, p, d_chi, d_K, d_K, d_s1, d_s2, d_s3);

    stretch(index_pow(4,k), p, d_chi_df, d_chi);
    gpu_mod_mul(n*n*p, d_gamma, d_chi, d_L);
    stretch(index_pow(4,k), p, d_chi_cd, d_chi);
    gpu_strassen(0, k, p, d_chi, d_L, d_L, d_s1, d_s2, d_s3);

    metric_pop(m_gpu_eval_detail, "HKL  ");

    // Compute the matrices A, B, C
    F *d_A = d_H;  
    F *d_B = d_K;  
    F *d_C = d_L;  

    metric_push(m_gpu_eval_detail);

    stretch(index_pow(4,k), p, d_chi_ad, d_chi);
    gpu_mod_mul(n*n*p, d_H, d_chi, d_A);
    stretch(index_pow(4,k), p, d_chi_bd, d_chi);
    gpu_strassen(1, k, p, d_A, d_chi, d_A, d_s1, d_s2, d_s3);

    stretch(index_pow(4,k), p, d_chi_be, d_chi);
    gpu_mod_mul(n*n*p, d_K, d_chi, d_B);
    stretch(index_pow(4,k), p, d_chi_ce, d_chi);
    gpu_strassen(1, k, p, d_B, d_chi, d_B, d_s1, d_s2, d_s3);

    stretch(index_pow(4,k), p, d_chi_cf, d_chi);
    gpu_mod_mul(n*n*p, d_L, d_chi, d_C);
    stretch(index_pow(4,k), p, d_chi_af, d_chi);
    gpu_strassen(1, k, p, d_chi, d_C, d_C, d_s1, d_s2, d_s3);

    metric_pop(m_gpu_eval_detail, "ABC  ");

    // Compute the matrix Q

    F *d_Q = d_B;

    metric_push(m_gpu_eval_detail);

    stretch(index_pow(4,k), p, d_chi_bc, d_chi);
    gpu_mod_mul(n*n*p, d_B, d_chi, d_B);
    stretch(index_pow(4,k), p, d_chi_ac, d_chi);
    gpu_mod_mul(n*n*p, d_C, d_chi, d_C);

    gpu_strassen(1, k, p, d_C, d_B, d_Q, d_s1, d_s2, d_s3);
    
    stretch(index_pow(4,k), p, d_chi_ab, d_chi);
    gpu_mod_mul(n*n*p, d_chi, d_Q, d_Q);
    gpu_mod_mul(n*n*p, d_Q, d_A, d_A);

    // Transpose data before reduction

    transpose(n*n*p,
              p,
              n*n,
              n*n,
              1,
              d_A,
              d_Q,
              true);

    gpu_mod_add_reduce(p, n*n, d_Q, d_chi);

    metric_pop(m_gpu_eval_detail, "Qre  ");

    dev_array_download(p, d_chi, Px0);

    // Copy result
    for(index_t i = 0; i < p_in; i++)
        Px0_in[i] = Px0[i];

    array_delete(Px0);

    metric_pop(m_gpu_eval, 
               "p = %4ld, n = %4ld, R = %8ld, N = %9ld, first = %9ld", 
               p, n, index_pow(7, k), 3*index_pow(7, k)-2, first);

}


template <typename F>
int work(int argc, char **argv) 
{
    int status = 0;
    scalar_t modulus = F::characteristic;

    if(!strcmp(argv[1], "create")) {
        assert(argc >= 6);
        index_t k = atol(argv[3]);
        index_t n = 1L << k;
        index_t seed = atol(argv[4]);
        srand(seed);
        const char *fn = argv[5];
        std::printf("create: modulus = %u, k = %ld, n = %ld, seed = %ld, instance-file = %s\n", modulus, k, n, seed, fn);

        F *chi = array_allocate<F>(15*n*n);

        // Prepare a random input instance
        array_rand(15*n*n, chi);

        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("R = %ld, N = %ld\n", 
                    R, N);
        assert(N <= modulus);

        // Write the instance to disk
        FILE *out;
        assert((out = fopen(fn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&k, sizeof(index_t), 1, out) == 1);
        assert(fwrite(chi, sizeof(F), 15*n*n, out) == 15*n*n);        
        assert(fclose(out) == 0);

        std::printf("wrote \"%s\"\n", fn);

        array_delete(chi);
    }
    if(!strcmp(argv[1], "evaluate")) {
        assert(argc >= 7);

        const char *infn = argv[2];
        index_t first = atol(argv[3]);
        index_t points = atol(argv[4]);
        index_t batch = atol(argv[5]);
        const char *outfn = argv[6];
        std::printf("evaluate: instance-file = %s, first = %ld, points = %ld, batch = %ld, eval-file = %s\n", infn, first, points, batch, outfn);
        
        // Read the instance from disk
        FILE *in;
        scalar_t in_modulus;
        index_t k;
        assert((in = fopen(infn, "r")) != NULL);
        assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
        assert(fread(&k, sizeof(index_t), 1, in) == 1);
        index_t n = 1L << k;
        std::printf("evaluate: modulus = %u, k = %ld, n = %ld\n", modulus, k, n);
        F *chi = array_allocate<F>(15*n*n);
        assert(fread(chi, sizeof(F), 15*n*n, in) == 15*n*n);
        assert(fclose(in) == 0);

        std::printf("read \"%s\"\n", infn);

        // Run the evaluation (and time it)

        F *x = array_allocate<F>(points);
        F *y = array_allocate<F>(points);        

        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("R = %ld, N = %ld\n", 
                    R, N);
        assert(N <= modulus);
        assert(first + points <= 4*R); // was <= N
        assert(first >= 0);

        coeff_tab<F> pre;
        gpu_coeff_precompute<F>(k, pre);

        for(index_t j = 0; j < points; j++)
            x[j] = F(first + j);

        // Upload chi matrices in interleaved form
        F *chi_i = array_allocate<F>(15*n*n);
        for(index_t i = 0; i < 15; i++) {
            for(index_t j = 0; j < n*n; j++) {
                index_t jj = to_interleaved(k, j);
                chi_i[i*n*n+jj] = chi[i*n*n+j];
            }
        }
        F *d_chi_base = dev_array_allocate<F>(15*n*n);
        dev_array_upload(15*n*n, chi_i, d_chi_base);
        array_delete(chi_i);

        index_t p = batch;
        index_t R_round = ((R+8192-1)/8192)*8192;
        index_t p_round = ((p+8192-1)/8192)*8192;
        index_t cap  = 4*n*n*p + p_round + p*R_round + p*(4*(R/7)+4*4*((R/7)/7));
        index_t cap2 = 4*n*n*p + 3*4*index_pow(7,k-2)*p*4;
        if(cap2 > cap)
            cap = cap2;
        F *d_scratch = dev_array_allocate<F>(cap);

        metric_push(m_default);
        for(index_t i = 0; i < (points+batch-1)/batch; i++)
            gpu_binom62_linear_poly((i+1)*batch < points ? batch : points-i*batch,
                                    k,
                                    pre,
                                    d_chi_base, 
                                    first + i*batch, 
                                    y + i*batch,
                                    d_scratch);
        metric_pop(m_default, "time");

        dev_array_delete(d_scratch);

        dev_array_delete(d_chi_base);

        gpu_coeff_release<F>(pre);

        // Write the evaluations to disk
        FILE *out;
        assert((out = fopen(outfn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&points, sizeof(index_t), 1, out) == 1);
        assert(fwrite(x, sizeof(F), points, out) == points);        
        assert(fwrite(y, sizeof(F), points, out) == points);        
        assert(fclose(out) == 0);

        array_delete(y);
        array_delete(x);
        array_delete(chi);
    }
    if(!strcmp(argv[1], "cat")) {
        assert(argc >= 4);
        const char *outfn = argv[argc-1];
        std::printf("interpolate: modulus = %u, out-eval-file = %s\n", 
                    modulus, outfn);

        index_t cat_in = 0;
        for(index_t j = 2; j < argc-1; j++) {
            const char *infn = argv[j];
            FILE *in;
            scalar_t in_modulus;
            index_t points;
            assert((in = fopen(infn, "r")) != NULL);
            assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
            assert(fread(&points, sizeof(index_t), 1, in) == 1);
            assert(fclose(in) == 0);
            cat_in += points;
        }
        std::printf("%ld file(s) with %ld point(s)\n", argc-3, cat_in);

        F *x_in = array_allocate<F>(cat_in);
        F *y_in = array_allocate<F>(cat_in);

        index_t total_points = 0;
        for(index_t j = 2; j < argc-1; j++) {
            const char *infn = argv[j];
            // Read input into x_in and y_in

            FILE *in;
            scalar_t in_modulus;
            index_t points;
            assert((in = fopen(infn, "r")) != NULL);
            assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
            assert(fread(&points, sizeof(index_t), 1, in) == 1);
            assert(total_points + points <= cat_in);
            assert(fread(x_in + total_points, sizeof(F), points, in) == points);        
            assert(fread(y_in + total_points, sizeof(F), points, in) == points);        
            assert(fclose(in) == 0);

            std::printf("read \"%s\" (%ld points)\n", infn, points);

            total_points += points;
        }
        assert(total_points == cat_in);

        // Write the concatenated evaluations to disk
        FILE *out;
        assert((out = fopen(outfn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&cat_in, sizeof(index_t), 1, out) == 1);
        assert(fwrite(x_in, sizeof(F), cat_in, out) == cat_in);
        assert(fwrite(y_in, sizeof(F), cat_in, out) == cat_in);        
        assert(fclose(out) == 0);
        
        std::printf("wrote \"%s\" (%ld points)\n", outfn, cat_in);

        array_delete(y_in);
        array_delete(x_in);
    }
    if(!strcmp(argv[1], "interpolate")) {
        assert(argc >= 5);

        // Interpolation assumes exact number points sufficient for interpolation

        index_t k = atol(argv[2]);
        const char *outfn = argv[argc-1];
        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("interpolate: modulus = %u, k = %ld, R = %ld, N = %ld, proof-file = %s\n", 
                    modulus, k, R, N, outfn);
        assert(N <= modulus);

        F *x = array_allocate<F>(N);
        F *y = array_allocate<F>(N);        

        index_t total_points = 0;
        for(index_t j = 3; j < argc-1; j++) {
            const char *infn = argv[j];
            // Read input into x and y

            FILE *in;
            scalar_t in_modulus;
            index_t points;
            assert((in = fopen(infn, "r")) != NULL);
            assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
            assert(fread(&points, sizeof(index_t), 1, in) == 1);
            assert(total_points + points <= N);
            assert(fread(x + total_points, sizeof(F), points, in) == points);        
            assert(fread(y + total_points, sizeof(F), points, in) == points);        
            assert(fclose(in) == 0);

            std::printf("read \"%s\" (%ld points)\n", infn, points);

            total_points += points;
        }
        assert(total_points == N);

        F b0 = F(0);
        for(index_t i = 0; i < N; i++)
            if(x[i].value() < R)
                b0 = b0 + y[i];

        metric_push(m_default);
        Poly<F> p = Poly<F>::interpolate(N, x, y);
        metric_pop(m_default, "time");

        std::printf("value: %11u\n", (uint) b0.value());

        F *pa = array_allocate<F>(N);
        index_t d = p.degree();
        for(index_t i = 0; i < N; i++)
            pa[i] = i <= d ? p[i] : F(0);

        // Write the proof polynomial to disk
        FILE *out;
        assert((out = fopen(outfn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&N, sizeof(index_t), 1, out) == 1);
        assert(fwrite(pa, sizeof(F), N, out) == N);    
        assert(fclose(out) == 0);
        std::printf("wrote \"%s\" (%ld points)\n", outfn, N);

        array_delete(pa);
        array_delete(y);
        array_delete(x);
    }
    if(!strcmp(argv[1], "corrupt")) {
        assert(argc == 6);
        index_t corrupt_points = atol(argv[2]);
        index_t seed = atol(argv[3]);
        srand(seed);
        const char *infn = argv[4];
        const char *outfn = argv[5];
        std::printf("corrupt: modulus = %u, corrupt-points = %ld, seed = %ld, in-file = %s, out-file = %s\n", 
                    modulus, corrupt_points, seed, infn, outfn);

        FILE *in;
        scalar_t in_modulus;
        index_t points;
        assert((in = fopen(infn, "r")) != NULL);
        assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
        assert(fread(&points, sizeof(index_t), 1, in) == 1);
        assert(points >= 0);

        F *x = array_allocate<F>(points);
        F *y = array_allocate<F>(points);

        assert(fread(x, sizeof(F), points, in) == points);        
        assert(fread(y, sizeof(F), points, in) == points);        
        assert(fclose(in) == 0);

        std::printf("read %ld point(s) from %s\n", points, infn);

        index_t *p = array_allocate<index_t>(points);
        randperm(points, p);
        assert(corrupt_points <= points);
        for(index_t i = 0; i < corrupt_points; i++) {
            F was = y[p[i]];
            F offset = F(0);
            while(offset == F(0))
                offset = F::rand();
            F corrupt = was + offset;
            y[p[i]] = corrupt;
            std::printf("corrupting at x = %10u from y = %10u to y = %10u\n", 
                        x[p[i]].value(), was.value(), corrupt.value());
        }
        std::printf("corrupted %ld point(s) in total\n", corrupt_points);
        array_delete(p);

        // Write the corrupted evaluations to disk
        FILE *out;
        assert((out = fopen(outfn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&points, sizeof(index_t), 1, out) == 1);
        assert(fwrite(x, sizeof(F), points, out) == points);
        assert(fwrite(y, sizeof(F), points, out) == points);        
        assert(fclose(out) == 0);

        std::printf("wrote \"%s\" (%ld points)\n", outfn, points);

        array_delete(y);
        array_delete(x);
    }
    if(!strcmp(argv[1], "decode")) {
        assert(argc >= 5);

        // Decoding tries to interpolate from potentially erroneous data

        index_t k = atol(argv[2]);
        const char *outfn = argv[argc-1];
        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("decode: modulus = %u, k = %ld, R = %ld, N = %ld, proof-file = %s\n", 
                    modulus, k, R, N, outfn);
        assert(N <= modulus);

        // Find out how many points we have in the evaluation files supplied as input

        index_t E_in = 0;

        for(index_t j = 3; j < argc-1; j++) {
            const char *infn = argv[j];
            FILE *in;
            scalar_t in_modulus;
            index_t points;
            assert((in = fopen(infn, "r")) != NULL);
            assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
            assert(fread(&points, sizeof(index_t), 1, in) == 1);
            assert(fclose(in) == 0);
            E_in += points;
        }
        std::printf("%ld file(s) with E = %ld point(s), need N = %ld point(s)\n", argc-4, E_in, N);

        assert(E_in >= N);

        F *x_in = array_allocate<F>(E_in);
        F *y_in = array_allocate<F>(E_in);
        index_t *s_in = array_allocate<index_t>(E_in);

        index_t total_points = 0;
        for(index_t j = 3; j < argc-1; j++) {
            const char *infn = argv[j];
            // Read input into x_in and y_in

            FILE *in;
            scalar_t in_modulus;
            index_t points;
            assert((in = fopen(infn, "r")) != NULL);
            assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
            assert(fread(&points, sizeof(index_t), 1, in) == 1);
            assert(total_points + points <= E_in);
            assert(fread(x_in + total_points, sizeof(F), points, in) == points);        
            assert(fread(y_in + total_points, sizeof(F), points, in) == points);        
            assert(fclose(in) == 0);

            std::printf("read \"%s\" (%ld points)\n", infn, points);

            total_points += points;
        }
        assert(total_points == E_in);


        for(index_t i = 0; i < E_in; i++)
            s_in[i] = i;
        heapsort_indirect(E_in, x_in, s_in);

        index_t E = 0;
        for(index_t i = 1; i < E_in; i++) {
            assert(x_in[s_in[i-1]] <= x_in[s_in[i]]);
            if(x_in[s_in[i-1]] < x_in[s_in[i]])
                s_in[++E] = s_in[i];
        }
        if(E_in > 0)
            E++;
        std::printf("out of the %ld points %ld are distinct, updating E = %ld\n", E_in, E, E);
        assert(E >= N);

        F *x = array_allocate<F>(E);
        F *y = array_allocate<F>(E);
        for(index_t i = 0; i < E; i++) {
            x[i] = x_in[s_in[i]];
            y[i] = y_in[s_in[i]];
        }

        array_delete(x_in);
        array_delete(y_in);
        array_delete(s_in);
                   
        metric_push(m_default);
        Poly<F> p;
        bool success = rs_decode_xy(N-1, E, x, y, p);
        metric_pop(m_default, "time");
        if(success)
            std::printf("decoding successful\n");
        else
            std::printf("DECODING FAILURE\n");

        assert(success);

        F *xr = array_allocate<F>(R);
        F *yr = array_allocate<F>(R);
        for(index_t i = 0; i < R; i++)
            xr[i] = F(i);
        metric_push(m_default);
        p.batch_eval(R, xr, yr);
        metric_pop(m_default, "batch_eval1");
        
        F b0 = F(0);
        for(index_t i = 0; i < R; i++)
            b0 = b0 + yr[i];
        std::printf("value: %11u\n", (uint) b0.value());

        array_delete(yr);
        array_delete(xr);

        F *yc = array_allocate<F>(E);
        metric_push(m_default);
        p.batch_eval(E, x, yc);
        metric_pop(m_default, "batch_eval2");
        index_t error_count = 0;
        for(index_t i = 0; i < E; i++)
            if(y[i] != yc[i])
                error_count++;
        if(error_count == 0) {
            std::printf("all evaluations agree with the decoded polynomial\n");
        } else {
            std::printf("%ld evaluation(s) disagree with the decoded polynomial:\n", 
                        error_count);
            for(index_t i = 0; i < E; i++)
                if(y[i] != yc[i])
                    std::printf("x = %10u has y = %10u whereas p(x) = %10u\n",
                                x[i].value(), y[i].value(), yc[i].value());
        }
        array_delete(yc);

        F *pa = array_allocate<F>(N);
        index_t d = p.degree();
        for(index_t i = 0; i < N; i++)
            pa[i] = i <= d ? p[i] : F(0);

        // Write the proof polynomial to disk
        FILE *out;
        assert((out = fopen(outfn, "w")) != NULL);
        assert(fwrite(&modulus, sizeof(scalar_t), 1, out) == 1);
        assert(fwrite(&N, sizeof(index_t), 1, out) == 1);
        assert(fwrite(pa, sizeof(F), N, out) == N);    
        assert(fclose(out) == 0);
        std::printf("wrote \"%s\" (%ld points)\n", outfn, N);

        array_delete(pa);
        array_delete(y);
        array_delete(x);
    }
    if(!strcmp(argv[1], "verify")) {
        assert(argc >= 6);

        index_t points = atol(argv[2]);
        index_t seed = atol(argv[3]);
        srand(seed);
        const char *infn = argv[4];
        const char *pffn = argv[5];

        std::printf("verify: modulus = %u, number-of-points = %ld, seed = %ld, instance-file = %s, proof-file = %s\n",
                    modulus, points, seed, infn, pffn);

        // Read the instance from disk
        FILE *in;
        scalar_t in_modulus;
        index_t k;
        assert((in = fopen(infn, "r")) != NULL);
        assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
        assert(fread(&k, sizeof(index_t), 1, in) == 1);
        index_t n = 1L << k;
        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("verify: modulus = %u, k = %ld, n = %ld, R = %ld, N = %ld\n", 
                    modulus, k, n, R, N);
        F *chi = array_allocate<F>(15*n*n);
        assert(fread(chi, sizeof(F), 15*n*n, in) == 15*n*n);        
        assert(fclose(in) == 0);

        std::printf("read \"%s\"\n", infn);

        // Read the proof polynomial from disk
        F *pa = array_allocate<F>(N);      
        assert((in = fopen(pffn, "r")) != NULL);
        assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
        index_t NN;
        assert(fread(&NN, sizeof(index_t), 1, in) == 1);
        assert(N == NN);
        assert(fread(pa, sizeof(F), N, in) == N);        
        assert(fclose(in) == 0);
        std::printf("read \"%s\" (%ld points)\n", pffn, N);
        Poly<F> p = Poly<F>::x(N-1);
        for(index_t i = 0; i < N; i++)
            p[i] = pa[i];
        array_delete(pa);

        // Check the proof at random points
        metric_push(m_default);
        for(index_t r = 0; r < points; r++) {
            F x0  = F::rand();
            F y0  = p.eval(x0);
            F y0p = binom62_linear_poly(k, chi, x0);
            index_t ok = y0 == y0p;
            std::printf("x0 = %11u, y0 = %11u, y0p = %11u", 
                        (uint) x0.value(), 
                        (uint) y0.value(), 
                        (uint) y0p.value());
            if(!ok) {
                std::printf(" -- VERIFICATION FAILURE");
                status = 1;
            }
            std::printf("\n");
        }
        metric_pop(m_default, "time");

        array_delete(chi);
    }

    if(!strcmp(argv[1], "force")) {
        assert(argc >= 3);

        const char *infn = argv[2];
        std::printf("force: instance-file = %s\n",
                    infn);

        // Read the instance from disk
        FILE *in;
        scalar_t in_modulus;
        index_t k;
        assert((in = fopen(infn, "r")) != NULL);
        assert(fread(&in_modulus, sizeof(scalar_t), 1, in) == 1 && modulus == in_modulus);
        assert(fread(&k, sizeof(index_t), 1, in) == 1);
        index_t n = 1L << k;
        index_t R = index_pow(7, k);
        index_t N = 3*R-2;
        std::printf("force: modulus = %u, k = %ld, n = %ld, R = %ld, N = %ld\n", 
                    modulus, k, n, R, N);
        F *chi = array_allocate<F>(15*n*n);
        assert(fread(chi, sizeof(F), 15*n*n, in) == 15*n*n);        
        assert(fclose(in) == 0);

        std::printf("read \"%s\"\n", infn);

        metric_push(m_default);
        F b1 = binom62_linear(k, chi);
        metric_pop(m_default, "time");
        std::printf("value: %11u\n", (uint) b1.value());

        array_delete(chi);
    }
    if(!strcmp(argv[1], "test")) {
        std::printf("test: modulus = %u\n", modulus);
        test_F<F>();
    }

    return status;
}

/****************************************************** Program entry point. */

int main(int argc, char **argv)
{
    srand(1234567);

    std::printf("host: %s\n", sysdep_hostname());
    std::printf("invoked as:");
    for(index_t i = 0; i < argc; i++)
        std::printf(" %s", argv[i]);
    std::printf("\n");

    if(argc <= 1) {
        std::printf("usage:\n"
                    "%s create      [modulus] [k] [random-seed] [instance-file]\n"
                    "%s force       [instance-file]\n"
                    "%s evaluate    [instance-file] [first] [num-of-points] [batch-size] [eval-file]\n"
                    "%s cat         [eval-file(s)] [out-eval-file]\n"
                    "%s interpolate [k] [eval-file(s)] [proof-file]\n"
                    "%s corrupt     [num-of-points] [random-seed] [eval-file] [corrupted-file]\n"
                    "%s decode      [k] [eval-file(s)] [proof-file]\n"
                    "%s verify      [num-of-points] [random-seed] [instance-file] [proof-file]\n"
                    "%s test        [modulus]"
                    "\n"
                    "available moduli: " MODULUS_LIST "\n",
           argv[0],
           argv[0],
           argv[0],
           argv[0],
           argv[0],
           argv[0],
           argv[0],
           argv[0],
           argv[0]);
        return 0;
    }

    bool got_command = false;
    scalar_t modulus = 0U;

    if(!strcmp(argv[1], "create") || 
       !strcmp(argv[1], "test")) {
        assert(argc >= 3);
        modulus = atol(argv[2]);
        got_command = true;     
    }
    if(!strcmp(argv[1], "force") ||
       !strcmp(argv[1], "evaluate") ||
       !strcmp(argv[1], "cat") ||
       !strcmp(argv[1], "interpolate") ||
       !strcmp(argv[1], "decode") ||
       !strcmp(argv[1], "corrupt") ||
       !strcmp(argv[1], "verify")) {
        index_t file_arg;
        if(!strcmp(argv[1], "force") ||
           !strcmp(argv[1], "evaluate") ||
           !strcmp(argv[1], "cat")) {
            file_arg = 2;
        }
        if(!strcmp(argv[1], "interpolate") ||
           !strcmp(argv[1], "decode")) {
            file_arg = 3;
        }
        if(!strcmp(argv[1], "verify") ||
           !strcmp(argv[1], "corrupt")) {
            file_arg = 4;
        }
        assert(argc >= file_arg + 1);
        // get the modulus from instance or evaluation file
        FILE *in; 
        assert((in = fopen(argv[file_arg], "r")) != NULL);
        assert(fread(&modulus, sizeof(scalar_t), 1, in) == 1);
        assert(fclose(in) == 0);
        got_command = true;     
    }

    if(!got_command) {
        std::printf("error: unrecognised command\n");
        return 1;
    }

    int status = work_arbiter(modulus, argc, argv);
    if(status != 0) {
        std::printf("FAILURE\n");
    }

    assert(alloc_balance == 0);
    assert(dev_alloc_balance == 0);
    assert(start_stack_top == -1);
    assert(memtrack_stack_top == -1);
    assert(dev_memtrack_stack_top == -1);
    return status;
}
