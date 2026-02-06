#pragma once

#include <dlpack/dlpack.h>

#include <concepts>
#include <ostream>
#include <sstream>
#include <utility>

#if __cplusplus >= 202002L && __has_include(<source_location>)
#include <source_location>
#define HAS_SOURCE_LOCATION 1
#else
#define HAS_SOURCE_LOCATION 0
#endif

namespace host {

struct PanicError : public std::runtime_error {
 public:
  // copy and move constructors
  explicit PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto detail() const -> std::string_view {
    const auto sv = std::string_view{m_message};
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

 private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]]
inline auto panic(
#if HAS_SOURCE_LOCATION
    std::source_location location
#else
    const char* file = __builtin_FILE(), int line = __builtin_LINE()
#endif
    , Args&&... args) -> void {
  std::ostringstream os;
#if HAS_SOURCE_LOCATION
  os << "Runtime check failed at " << location.file_name() << ":" << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
#else
  os << "Runtime check failed at " << file << ":" << line;
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  }
#endif
  throw PanicError(std::move(os).str());
}

template <typename... Args>
struct RuntimeCheck {
#if HAS_SOURCE_LOCATION
  using Loc_t = std::source_location;
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, Loc_t location = Loc_t::current()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(location, std::forward<Args>(args)...);
    }
  }
#else
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, const char* file = __builtin_FILE(), int line = __builtin_LINE()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(file, line, std::forward<Args>(args)...);
    }
  }
#endif
};

template <typename Cond, typename... Args>
explicit RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <std::signed_integral T, std::signed_integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

template <std::unsigned_integral T, std::unsigned_integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
inline auto offset(T* ptr, U... offset) -> void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char*>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
inline auto offset(const T* ptr, U... offset) -> const void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char*>(ptr) + (... + offset);
}

}  // namespace pointer

}  // namespace host