#include "wrapper/include/substruct_library.h"

#include <GraphMol/SmilesParse/SmilesParse.h>

#include <memory>
#include <string>

namespace smarts_evolution {

namespace {

std::string to_string(rust::Str value) {
  return std::string(value.data(), value.size());
}

}  // namespace

CompiledSmartsQuery::CompiledSmartsQuery(std::unique_ptr<RDKit::RWMol> query)
    : query_(std::move(query)) {}

const RDKit::ROMol &CompiledSmartsQuery::query() const { return *query_; }

BinaryClassificationLibrary::BinaryClassificationLibrary()
    : positive_holder_(boost::make_shared<RDKit::CachedTrustedSmilesMolHolder>()),
      negative_holder_(boost::make_shared<RDKit::CachedTrustedSmilesMolHolder>()),
      positive_patterns_(boost::make_shared<RDKit::PatternHolder>()),
      negative_patterns_(boost::make_shared<RDKit::PatternHolder>()),
      positive_library_(positive_holder_, positive_patterns_),
      negative_library_(negative_holder_, negative_patterns_) {}

bool BinaryClassificationLibrary::add_smiles(rust::Str smiles, bool positive) {
  std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(to_string(smiles)));
  if (!mol) {
    return false;
  }

  if (positive) {
    positive_library_.addMol(*mol);
  } else {
    negative_library_.addMol(*mol);
  }

  return true;
}

bool BinaryClassificationLibrary::count_matches(rust::Str smarts,
                                                std::int32_t num_threads,
                                                std::uint64_t &tp,
                                                std::uint64_t &fp,
                                                std::uint64_t &tn,
                                                std::uint64_t &fn_) const {
  std::unique_ptr<RDKit::RWMol> query(RDKit::SmartsToMol(to_string(smarts)));
  if (!query) {
    return false;
  }

  count_matches_compiled(CompiledSmartsQuery(std::move(query)), num_threads, tp, fp,
                         tn, fn_);
  return true;
}

void BinaryClassificationLibrary::count_matches_compiled(
    const CompiledSmartsQuery &query, std::int32_t num_threads,
    std::uint64_t &tp, std::uint64_t &fp, std::uint64_t &tn,
    std::uint64_t &fn_) const {
  RDKit::SubstructMatchParameters params;
  const auto positive_count = positive_size();
  const auto negative_count = negative_size();

  tp = positive_count == 0
           ? 0
           : static_cast<std::uint64_t>(
                 positive_library_.countMatches(query.query(), params, num_threads));
  fp = negative_count == 0
           ? 0
           : static_cast<std::uint64_t>(
                 negative_library_.countMatches(query.query(), params, num_threads));
  fn_ = positive_count - tp;
  tn = negative_count - fp;
}

rust::Vec<std::uint32_t> BinaryClassificationLibrary::positive_matches(
    rust::Str smarts, std::int32_t num_threads) const {
  rust::Vec<std::uint32_t> indices;
  if (positive_size() == 0) {
    return indices;
  }

  std::unique_ptr<RDKit::RWMol> query(RDKit::SmartsToMol(to_string(smarts)));
  if (!query) {
    return indices;
  }

  RDKit::SubstructMatchParameters params;
  for (auto match_idx : positive_library_.getMatches(*query, params, num_threads)) {
    indices.push_back(static_cast<std::uint32_t>(match_idx));
  }
  return indices;
}

std::size_t BinaryClassificationLibrary::positive_size() const {
  return positive_holder_->size();
}

std::size_t BinaryClassificationLibrary::negative_size() const {
  return negative_holder_->size();
}

std::unique_ptr<BinaryClassificationLibrary> new_binary_classification_library() {
  return std::make_unique<BinaryClassificationLibrary>();
}

std::unique_ptr<CompiledSmartsQuery> new_compiled_smarts_query(rust::Str smarts) {
  std::unique_ptr<RDKit::RWMol> query(RDKit::SmartsToMol(to_string(smarts)));
  if (!query) {
    return nullptr;
  }
  return std::make_unique<CompiledSmartsQuery>(std::move(query));
}

}  // namespace smarts_evolution
