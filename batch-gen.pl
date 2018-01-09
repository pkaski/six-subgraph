
# Miscellaneous batch generation

# Ten least k's for which 2^n-k is prime

# n = 31; k = 1, 19, 61, 69, 85, 99, 105, 151, 159, 171 

#@list=(1,19,61,69,85,99,105,151,159,171);

@list=(19,61,69);


sub normalize {
    my $x = $_[0];
    my $m = $_[1];
    my $r = $x % $m;
    if($r < 0) {
        $r = $m + $r; 
    }
    return $r;
}

sub first_bezout_factor {
    my $a = $_[0];
    my $b = $_[1];
    my $old_s = 1;
    my $old_t = 0;
    my $old_r = $a;
    my $s = 0;
    my $t = 1;
    my $r = $b;
    while($r != 0) {
        my $z = $old_r % $r;
	my $q = ($old_r - $z)/$r;
	die "bad division" unless $q*$r + $z == $old_r;

        my $save = $old_r;
        $old_r = $r;
        $r = $save - $q*$r;
	die "bad division" unless $q*$old_r + $r == $save;

        $save = $old_s;
        $old_s = $s;
        $s = $save - $q*$s;
	die "bad division" unless $q*$old_s + $s == $save;

        $save = $old_t;
        $old_t = $t;
        $t = $save - $q*$t;     
	die "bad division" unless $q*$old_t + $t == $save;
	
    }
    die "not coprime" unless $old_r == 1;
    die "bad bezout coefficients" unless $a*$old_s + $b*$old_t == $old_r;
    return $old_s;
}


my $prime_template = <<'END_TEMPLATE';
struct P__PRIME__U
{
    static const scalar_t modulus         = __PRIME__U;
    static const scalar_t overflow_adjust = __OVERFLOW_ADJUST__U;
      // (not needed for 31-bit or shorter primes)
      // = 2^{32} mod MODULUS
    static const scalar_t montgomery_R    = __MONTGOMERY_R__U;
      // = mod_reduce(1L << 32);
    static const scalar_t montgomery_R2   = __MONTGOMERY_R2__U;
      // = mod_mul(mod_reduce(1L << 32), mod_reduce(1L << 32));
    static const scalar_t montgomery_F    = __MONTGOMERY_F__U;
      // = normalize(-first_bezout_factor(MODULUS, 1 << 32), 1L << 32);
};

__device__ inline 
scalar_t mod_mul_montgomery_ptx_P__PRIME__U(scalar_t x, scalar_t y)
{
    scalar_t r;
    asm("{\n\t"
        "   .reg .u32  Tl;                  \n\t"
        "   .reg .u32  Th;                  \n\t"
        "   .reg .u32  m;                   \n\t"
        "   .reg .u32  tl;                  \n\t"
        "   .reg .u32  th;                  \n\t"
        "   .reg .u32  Mf;                  \n\t"
        "   .reg .u32  Mo;                  \n\t"
        "   .reg .pred p;                   \n\t"
        "   mov.u32         Mo, __PRIME__U; \n\t"
        "   mov.u32         Mf, __MONTGOMERY_F__U; \n\t"
        "   mul.lo.u32      Tl, %1, %2;     \n\t"
        "   mul.hi.u32      Th, %1, %2;     \n\t"
        "   mul.lo.u32      m, Tl, Mf;      \n\t"
        "   mad.lo.cc.u32   tl, m, Mo, Tl;  \n\t"
        "   madc.hi.u32     th, m, Mo, Th;  \n\t"
        "   setp.ge.u32     p, th, Mo;      \n\t"
        "@p sub.u32         th, th, Mo;     \n\t"
        "   mov.u32         %0, th;         \n\t"
        "}\n\t"
        : // output [and input--output] operands below
          // '=' write-register
          // '+' read-and-write register
          // 'r' .u32 reg
          // 'l' .u64 reg
        "=r"(r)
        : // input operands below [omit colon if none]
        "r"(x),
        "r"(y)
        );
    return r;
}
END_TEMPLATE

$dev_mul_template =<<'END_TEMPLATE';
template <>
__host__ __device__ inline 
scalar_t mod_mul_montgomery<struct P__PRIME__U>(scalar_t x, scalar_t y)
{
#ifdef __CUDA_ARCH__
    return mod_mul_montgomery_ptx_P__PRIME__U(x,y);
#else
    return mod_mul_montgomery_host<struct P__PRIME__U>(x,y);
#endif
}
END_TEMPLATE

$lookup_template =<<'END_TEMPLATE';
END_TEMPLATE


for($i=0;$i<=$#list;$i++) {
    $p  = (1 << 31)-$list[$i];
    $oa = (1 << 32) % $p;      
    $r  = normalize(1 << 32, $p);
    $r2 = normalize($r*$r, $p);
    $f  = normalize(-first_bezout_factor($p, 1 << 32), 1 << 32);
    $inst = $prime_template;
    $inst=~s/__PRIME__/$p/g;    
    $inst=~s/__OVERFLOW_ADJUST__/$oa/g;    
    $inst=~s/__MONTGOMERY_F__/$f/g;    
    $inst=~s/__MONTGOMERY_R__/$r/g;    
    $inst=~s/__MONTGOMERY_R2__/$r2/g;    
    print "$inst\\\n";
}
print "\n";

for($i=0;$i<=$#list;$i++) {
    $p  = (1 << 31)-$list[$i];
    $inst = $dev_mul_template;
    $inst=~s/__PRIME__/$p/g;    
    print "$inst\\\n";
}
print "\n";

$modulus_list="";
for($i=0;$i<=$#list;$i++) {
    $p  = (1 << 31)-$list[$i];
    if($i > 0) {
	$modulus_list=$modulus_list.", ";
    }
    $modulus_list=$modulus_list.$p;
}
print "#define MODULUS_LIST \"$modulus_list\"\n\n";

print <<'END_TEMPLATE';
int work_arbiter(scalar_t modulus, int argc, char **argv) {
    switch(modulus) {
END_TEMPLATE
$modulus_list="";
for($i=0;$i<=$#list;$i++) {
    $p  = (1 << 31)-$list[$i];
    $tmp =<< 'END_TEMPLATE';
        case __PRIME__:
            return work<Zp<P__PRIME__U>>(argc, argv);
END_TEMPLATE
    $tmp=~s/__PRIME__/$p/g;
    print $tmp;
}
print <<'END_TEMPLATE';
        default:
            std::printf("error: modulus %u not available\n", modulus);
            return 1;
    }
    // control never reaches here
}


END_TEMPLATE

$vec[0]="x";
$vec[1]="y";
$vec[2]="z";
$vec[3]="w";

print "#define YATES_47_BATCH\\\n";
for($i = 0; $i < 4; $i++) {
    print "   ";
    for($v = 0; $v < 4; $v++) {
	$l = $vec[$v];
	print " s$i\.$l = F(0).raw();";
    }
    print "\\\n";
}
for($j = 0; $j < 7; $j++) {
    print "    ";
    for($i = 0; $i < 4; $i++) {
	for($s = -1; $s < 2; $s+=2) {
	    if($s == -1) {
		$action = "-";
	    } else {
		$action = "+";
	    }
	    printf "if(A$i$j == %2d) {", $s;
	    for($v = 0; $v < 4; $v++) {
		$l = $vec[$v];
		print " s$i\.$l = (F(s$i\.$l\, true) $action F(r$j\.$l\, true)).raw();";
	    }
	    print " }\\\n";
	}
    }
}
print "\n\n";

print "#define YATES_74_BATCH\\\n";
for($i = 0; $i < 7; $i++) {
    print "   ";
    for($v = 0; $v < 4; $v++) {
	$l = $vec[$v];
	print " s$i\.$l = F(0).raw();";
    }
    print "\\\n";
}
for($j = 0; $j < 4; $j++) {
    print "    ";
    for($i = 0; $i < 7; $i++) {
	for($s = -1; $s < 2; $s+=2) {
	    if($s == -1) {
		$action = "-";
	    } else {
		$action = "+";
	    }
	    printf "if(A$j$i == %2d) {", $s;
	    for($v = 0; $v < 4; $v++) {
		$l = $vec[$v];
		print " s$i\.$l = (F(s$i\.$l\, true) $action F(r$j\.$l\, true)).raw();";
	    }
	    print " }\\\n";
	}
    }
}
print "\n\n";


sub to_interleaved {
    my $k = 2;
    my $r = 0;
    my $u = shift(@_);
    for(my $i = 0; $i < 2*$k; $i++) {
        $r = $r | ((($u >> $i)&1)<<(2*($i%$k)+($i/$k)));
    }
    return $r;
}

sub from_interleaved {
    my $k = 2;
    my $r = 0;
    my $u = shift(@_);
    for(my $i = 0; $i < 2*$k; $i++) {
        $r = $r | ((($u >> $i)&1)<<($k*($i%2)+($i/2)));
    }
    return $r;
}


$row[0] = "0";
$row[1] = "1";
$row[2] = "2";
$row[3] = "3";

$col[0] = "x";
$col[1] = "y";
$col[2] = "z";
$col[3] = "w";

sub entry {
    my $i = shift(@_);
    my $j = shift(@_);
    my $e = $i*4+$j;
    my $ei = to_interleaved($e);
    return sprintf("%d",$ei/4).".".$col[$ei%4];
}

print "#define MUL_4X4_BATCH\\\n";
for($ii = 0; $ii < 4; $ii++) {
    for($jj = 0; $jj < 4; $jj++) {
	for($kk = 0; $kk < 4; $kk++) {
	    $z = from_interleaved(4*$ii+$jj);
	    $i = sprintf("%d",$z/4);
	    $j = $z%4;
	    $k = $kk;
	    if($k == 0) {
		print "    c".entry($i,$j)." = (F(a".entry($i,$k).", true)*F(b".entry($k,$j).", true)).raw();\\\n";
	    } else {
		print "    c".entry($i,$j)." = (F(c".entry($i,$j).
                                           ", true) + F(a".entry($i,$k).", true)*F(b".entry($k,$j).", true)).raw();\\\n";
	    }
	}
    }
    print "    d_c[z + $ii*grid_size] = c$ii;\\\n";      
}
print "\n\n";

print "#define MUL_4X4_RIGHT_TRANSPOSE_BATCH\\\n";
for($ii = 0; $ii < 4; $ii++) {
    for($jj = 0; $jj < 4; $jj++) {
	for($kk = 0; $kk < 4; $kk++) {
	    $z = from_interleaved(4*$ii+$jj);
	    $i = sprintf("%d",$z/4);
	    $j = $z%4;
	    $k = $kk;
	    if($k == 0) {
		print "    c".entry($i,$j)." = (F(a".entry($i,$k).", true)*F(b".entry($j,$k).", true)).raw();\\\n";
	    } else {
		print "    c".entry($i,$j)." = (F(c".entry($i,$j).
		                           ", true) + F(a".entry($i,$k).", true)*F(b".entry($j,$k).", true)).raw();\\\n";
	    }
	}
    }
    print "    d_c[z + $ii*grid_size] = c$ii;\\\n";      
}
print "\n\n";




