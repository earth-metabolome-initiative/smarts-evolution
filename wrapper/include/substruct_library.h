#pragma once

#include "rust/cxx.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdint>
#include <memory>

#include <GraphMol/SubstructLibrary/SubstructLibrary.h>

namespace smarts_evolution {

class CompiledSmartsQuery {
 public:
  explicit CompiledSmartsQuery(std::unique_ptr<RDKit::RWMol> query);

  const RDKit::ROMol &query() const;

 private:
  std::unique_ptr<RDKit::RWMol> query_;
};

class BinaryClassificationLibrary {
 public:
  BinaryClassificationLibrary();

  bool add_smiles(rust::Str smiles, bool positive);
  bool count_matches(rust::Str smarts, std::int32_t num_threads,
                     std::uint64_t &tp, std::uint64_t &fp, std::uint64_t &tn,
                     std::uint64_t &fn_) const;
  void count_matches_compiled(const CompiledSmartsQuery &query,
                              std::int32_t num_threads, std::uint64_t &tp,
                              std::uint64_t &fp, std::uint64_t &tn,
                              std::uint64_t &fn_) const;
  rust::Vec<std::uint32_t> positive_matches(rust::Str smarts,
                                            std::int32_t num_threads) const;
  std::size_t positive_size() const;
  std::size_t negative_size() const;

 private:
  boost::shared_ptr<RDKit::CachedTrustedSmilesMolHolder> positive_holder_;
  boost::shared_ptr<RDKit::CachedTrustedSmilesMolHolder> negative_holder_;
  boost::shared_ptr<RDKit::PatternHolder> positive_patterns_;
  boost::shared_ptr<RDKit::PatternHolder> negative_patterns_;
  RDKit::SubstructLibrary positive_library_;
  RDKit::SubstructLibrary negative_library_;
};

std::unique_ptr<BinaryClassificationLibrary> new_binary_classification_library();
std::unique_ptr<CompiledSmartsQuery> new_compiled_smarts_query(rust::Str smarts);

}  // namespace smarts_evolution
