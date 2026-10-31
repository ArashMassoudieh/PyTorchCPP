// Decoupled test of CheckpointMemoryBuffer::seekoff (Hydro/evaluation/inference_runner.cpp),
// copied verbatim rather than linked against, matching the existing precedent in
// Hydro/experiments/gistohq_sligo/test_frozen_inference_guard.cpp (which copies
// requireDevelopmentOnly the same way) -- CheckpointMemoryBuffer lives in an
// anonymous namespace in inference_runner.cpp and is not exported for direct
// linking. If inference_runner.cpp's seekoff implementation changes, this copy
// must be updated to match, or the test stops meaning anything.
//
// Covers two historically real bugs, both already fixed in the production file:
// (1) seekoff never implemented at all (tellg()/seekg() always failed --
//     LibTorch's zip-format checkpoint reader could not load any checkpoint);
// (2) the bounds check formed `begin_ + off` (or `baseOffset + off`) before
//     validating `off`, which is undefined behavior for an out-of-range,
//     parser-controlled offset. This file specifically exercises off_type's
//     min/max from multiple seek bases, which is where (2) actually manifests.
//
// Build with -fsanitize=undefined,address to have UBSan/ASan, not just the
// assertions, prove no UB fires. A negative-control build of the pre-fix
// (unchecked) version under the same sanitizers reproduces a real
// "signed integer overflow" report on the off_type::max case -- confirming
// this is a meaningful regression test, not merely exercising already-safe code.
#include <streambuf>
#include <istream>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
class CheckpointMemoryBuffer : public std::streambuf {
public:
    explicit CheckpointMemoryBuffer(const std::vector<std::uint8_t>& bytes) {
        begin_ = const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
        end_ = begin_ + bytes.size();
        setg(begin_, begin_, end_);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in)) return pos_type(off_type(-1));
        const off_type size = static_cast<off_type>(end_ - begin_);
        off_type baseOffset;
        switch (dir) {
            case std::ios_base::beg: baseOffset = 0; break;
            case std::ios_base::cur: baseOffset = static_cast<off_type>(gptr() - begin_); break;
            case std::ios_base::end: baseOffset = size; break;
            default: return pos_type(off_type(-1));
        }
        if (off < -baseOffset || off > size - baseOffset) return pos_type(off_type(-1));
        const off_type requested = baseOffset + off;
        char* newPos = begin_ + requested;
        setg(begin_, newPos, end_);
        return pos_type(requested);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }

private:
    char* begin_ = nullptr;
    char* end_ = nullptr;
};

class CheckpointMemoryStream : public std::istream {
public:
    explicit CheckpointMemoryStream(const std::vector<std::uint8_t>& bytes)
        : std::istream(nullptr), buffer_(bytes) { rdbuf(&buffer_); }

private:
    CheckpointMemoryBuffer buffer_;
};

using off_type = std::streambuf::off_type;
}  // namespace

int main() {
    std::vector<std::uint8_t> data(256);
    for (int i = 0; i < 256; ++i) data[i] = static_cast<std::uint8_t>(i);

    // Ordinary seek/read correctness (regression guard for bug (1)).
    {
        CheckpointMemoryStream s(data);
        assert(s.tellg() == 0);
        s.seekg(0, std::ios_base::end);
        assert(s.tellg() == 256);
        s.seekg(100, std::ios_base::beg);
        char c;
        s.read(&c, 1);
        assert(static_cast<unsigned char>(c) == 100);
        assert(s.tellg() == 101);
        s.seekg(-1, std::ios_base::cur);
        s.read(&c, 1);
        assert(static_cast<unsigned char>(c) == 100);
        s.clear();
        s.seekg(1000, std::ios_base::beg);
        assert(s.fail());
    }

    // Overflow-prone cases (regression guard for bug (2)): off at off_type's
    // exact min/max, from both a zero and a nonzero seek base. Each must be
    // cleanly rejected, not crash or silently misbehave under UBSan/ASan.
    {
        CheckpointMemoryStream s(data);
        s.seekg(std::numeric_limits<off_type>::max(), std::ios_base::beg);
        assert(s.fail());
    }
    {
        CheckpointMemoryStream s(data);
        s.seekg(0, std::ios_base::end);
        s.clear();
        s.seekg(std::numeric_limits<off_type>::max(), std::ios_base::cur);
        assert(s.fail());
    }
    {
        CheckpointMemoryStream s(data);
        s.seekg(std::numeric_limits<off_type>::min(), std::ios_base::cur);
        assert(s.fail());
    }
    {
        CheckpointMemoryStream s(data);
        s.seekg(0, std::ios_base::end);
        s.clear();
        s.seekg(std::numeric_limits<off_type>::min(), std::ios_base::cur);
        assert(s.fail());
    }

    return 0;
}
